#!/usr/bin/env python3
"""
scripts/gemma_worker.py
========================
Continuous background worker for Gemma README Markdown enrichment.

Features:
  1. Runs continuously in a while loop (no manual execution needed).
  2. Processes repositories in parallel using a ThreadPoolExecutor.
  3. Uses a thread-safe rate limiter to never exceed GEMINI_RPM_LIMIT.
  4. Detects rate limits (HTTP 429) or transient API overloads and sleeps/backs off until normal service resumes.
  5. Updates Supabase database tables directly.

Usage:
    python3 scripts/gemma_worker.py
"""

# Ensure the project root is on the path regardless of where we run from
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

from database import PostgreSQLConnector
from utils.gemma_client import generate_readme_markdown, rate_limiter
from utils.readme_processor import process_markdown

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [GemmaWorker] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gemma.worker")

# Configuration Constants
CONCURRENT_THREADS = 4           # Number of parallel tasks
FETCH_BATCH_SIZE = 20            # How many tasks to pull from DB at once
IDLE_SLEEP_SECONDS = 30          # Time to sleep when database has no pending tasks
RATE_LIMIT_COOLDOWN_SECONDS = 60 # Cooldown sleep when a 429 or API exhaustion occurs

def process_single_repo(repo_id: str, full_name: str, description: str, readme_summary: str, db) -> bool:
    """Processes a single repository's README formatting and saves it to the database."""
    try:
        source_text = readme_summary or description
        if not source_text or not source_text.strip():
            # If no content, update database with placeholder to prevent reprocessing loop
            logger.info(f"Skipping enrichment for {full_name}: no raw text content available.")
            conn = db.connect()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE Repo SET readme_markdown = %s, updated_at = NOW() WHERE repo_id = %s;",
                ("No README content available.", repo_id)
            )
            conn.commit()
            cursor.close()
            conn.close()
            return True

        # Clean source text
        clean_text = process_markdown(source_text).clean_text
        if not clean_text or not clean_text.strip():
            return False

        # Generate markdown using Gemma Client (which handles RPM limits internally)
        markdown_out = generate_readme_markdown(clean_text[:3000])

        if markdown_out:
            conn = db.connect()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE Repo SET readme_markdown = %s, updated_at = NOW() WHERE repo_id = %s;",
                (markdown_out, repo_id)
            )
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"✅ Generated and saved Markdown for: {full_name}")
            return True
        else:
            logger.warning(f"⚠️ Failed to generate Markdown for {full_name} (possible rate-limiting).")
            return False

    except Exception as exc:
        logger.error(f"❌ Error processing repository {full_name}: {exc}")
        return False


def main():
    logger.info("Initializing Gemma README enrichment worker daemon...")
    
    db = PostgreSQLConnector()
    if not db.enabled or not db.verify_connection():
        logger.error("Could not connect to the database. Verify DATABASE_URL in .env. Exiting.")
        return

    db.init_db()
    logger.info(f"Connected to Supabase. Configuration: GEMINI_RPM_LIMIT={rate_limiter.rpm_limit}")
    logger.info("Starting background worker loop. Press Ctrl+C to terminate.")

    global_cooldown = False

    while True:
        try:
            if global_cooldown:
                logger.info(f"Entering rate-limit cooldown sleep of {RATE_LIMIT_COOLDOWN_SECONDS} seconds...")
                time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)
                global_cooldown = False

            # 1. Fetch repositories with empty readme_markdown
            conn = db.connect()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT repo_id, full_name, description, readme_summary "
                "FROM Repo "
                "WHERE readme_markdown IS NULL OR readme_markdown = '' "
                "ORDER BY star_count DESC "
                "LIMIT %s;",
                (FETCH_BATCH_SIZE,)
            )
            repos = cursor.fetchall()
            cursor.close()
            conn.close()

            if not repos:
                logger.info(f"No pending repositories to process. Sleeping for {IDLE_SLEEP_SECONDS} seconds...")
                time.sleep(IDLE_SLEEP_SECONDS)
                continue

            logger.info(f"Acquired batch of {len(repos)} repositories to enrich in parallel.")

            # 2. Process batch in parallel using ThreadPoolExecutor
            # Threads will automatically block on the thread-safe GeminiRateLimiter lock
            # ensuring that your RPM limit is never exceeded.
            success_count = 0
            with ThreadPoolExecutor(max_workers=CONCURRENT_THREADS) as executor:
                futures = {
                    executor.submit(process_single_repo, r[0], r[1], r[2], r[3], db): r[1]
                    for r in repos
                }
                
                for future in as_completed(futures):
                    repo_name = futures[future]
                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                        else:
                            # A failure often indicates rate limits (HTTP 429) or timeouts
                            global_cooldown = True
                    except Exception as exc:
                        logger.error(f"Thread execution failed for {repo_name}: {exc}")
                        global_cooldown = True

            logger.info(f"Batch completed: {success_count}/{len(repos)} enriched successfully.")

        except KeyboardInterrupt:
            logger.info("Termination signal received. Exiting gracefully.")
            break
        except Exception as err:
            logger.exception(f"Unexpected worker loop exception: {err}")
            time.sleep(10)

if __name__ == "__main__":
    main()
