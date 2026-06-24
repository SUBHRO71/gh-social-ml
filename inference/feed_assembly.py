import math
import random
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class FeedAssemblySystem:
    @staticmethod
    def process_feed_assembly(candidates: List[Dict[str, Any]], target_size: int = 15) -> List[str]:
        """
        Executes Freshness Injection and Exploration Injection sequentially
        on the pre-ranked top-15 JSON payload from the ranking system.
        """
        current_time = datetime.now(timezone.utc)
        
        # --- PART 1: FRESHNESS INJECTION ---
        for item in candidates:
            if 'score' not in item or item['score'] is None:
                item['score'] = 0.5
                
            raw_created_at = item.get('created_at')
            if not raw_created_at:
                continue
                
            try:
                if isinstance(raw_created_at, str):
                    clean_timestamp = raw_created_at.replace('Z', '+00:00')
                    created_date = datetime.fromisoformat(clean_timestamp)
                elif isinstance(raw_created_at, datetime):
                    created_date = raw_created_at
                else:
                    continue

                if created_date.tzinfo is None:
                    created_date = created_date.replace(tzinfo=timezone.utc)

                # Calculate decimal age in hours
                age_hours = max(0.0, (current_time - created_date).total_seconds() / 3600.0)

                # Apply smooth hyperbolic logarithmic decay boost if under 48 hours old
                if age_hours < 48.0:
                    boost = 0.25 * (1.1 / (1.0 + math.log1p(age_hours)))
                    item['score'] += boost
            except Exception as e:
                logger.error(f"Freshness parsing failed for repo {item.get('repo_id')}: {e}")
                continue

        # Re-sort the 15 repos after applying freshness boosts
        candidates.sort(key=lambda x: x.get('score', 0.5), reverse=True)

        # --- PART 2: EXPLORATION INJECTION ---
        # Safeguard anchor tier (Top 10) and introduce discovery variations to the bottom tier (Bottom 5)
        if len(candidates) >= 5:
            exploit_tier = candidates[:10]
            explore_tier = candidates[10:]
            
            random.shuffle(explore_tier)
            final_pool = exploit_tier + explore_tier
        else:
            final_pool = candidates

        # Strip internal temporary scores and return clean ordered string IDs
        return [str(item['repo_id']) for item in final_pool[:target_size]]