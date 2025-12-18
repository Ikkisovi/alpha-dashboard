
import json
import os
import hashlib
from compute_service import compute_metrics

CACHE_DIR = "/tmp/alphasage_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

COMMON_QUERIES = [
    # Default view (most common)
    {
        "pool_id": "pool_100000.json",
        "target_horizons": [20],
        "strategy": {"type": "long_short", "top_pct": 0.2, "rebalance_days": 20}
    },
    # Multi-horizon analysis
    {
        "pool_id": "pool_100000.json",
        "target_horizons": [5, 10, 20],
        "strategy": {"type": "long_short", "top_pct": 0.2, "rebalance_days": 20}
    },
    # Long-only comparison
    {
        "pool_id": "pool_100000.json",
        "target_horizons": [20],
        "strategy": {"type": "long_only", "top_pct": 0.2, "rebalance_days": 20}
    }
]

def generate_cache_key(request):
    # Match the TS logic: 
    # parts = [pool_id, factor_ids_str, train_start, test_end, horizons_str, strat_type, top_pct, reb_days]
    # NOTE: We need to match defaults if they aren't in request.
    # For now, simplistic string dump hash
    s = json.dumps(request, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def warmup():
    print(f"Warming up cache in {CACHE_DIR}...")
    for query in COMMON_QUERIES:
        print(f"Compute: {json.dumps(query)}")
        try:
            result = compute_metrics(query)
            if "error" in result:
                print(f"Error: {result['error']}")
                continue
                
            key = generate_cache_key(query)
            p = os.path.join(CACHE_DIR, f"{key}.json")
            with open(p, 'w') as f:
                json.dump(result, f)
            print(f"Cached {key} ({len(result.get('metrics', []))} metrics)")
        except Exception as e:
            print(f"Failed: {e}")

if __name__ == "__main__":
    warmup()
