# Performance Debug Summary

## Problem
**Strategy Configuration** in the dashboard hits 60s timeout limit when computing portfolio returns.

---

## Current Architecture

### API Endpoints
| Endpoint | Script | Purpose | Timeout |
|----------|--------|---------|---------|
| `/api/v1/compute/metrics` | `compute_service.py` | Full factor evaluation + analytics | 60s |
| `/api/v1/strategy` | `strategy_service.py` | Lightweight strategy-only | 15s |

### Data Flow
```
Frontend (page.tsx)
    ↓
fetchDynamicMetrics() calls ApiClient.computeStrategy()
    ↓
/api/v1/strategy/route.ts ensures combined factor signal (cached .npy on disk)
    ↓
strategy_service.py reads binary close prices + cached signal
    ↓
Returns daily portfolio returns
```

---

## Changes Made

### 1. Created Lightweight Strategy Service
**File:** `python-backend/strategy_service.py`
- Reads close prices directly from binary `.day.bin` files
- Bypasses Qlib initialization completely
- Only computes portfolio returns (no IC, VIF, etc.)

### 2. Created New API Route
**File:** `app/api/v1/strategy/route.ts`
- Spawns `strategy_service.py` instead of `compute_service.py`
- 15s timeout

### 3. Signal cache for factor selections
**Files:** `app/api/v1/strategy/route.ts`, `python-backend/compute_service.py`, `python-backend/strategy_service.py`
- When the frontend supplies `pool_id` + `factor_ids`, the API route hashes those parameters and saves/loads a combined equal-weight signal from `tmp/strategy_signals/*.npz`.
- Each `.npz` now includes the combined signal, the 1-day target return tensor, and the exact factor-date index, so the lightweight service can backtest purely from cached tensors (no binary close-price reads).
- Cache misses invoke `compute_service.py` once with `skip_analytics: true` and `combined_signal_output` so subsequent strategy tweaks stay fast. This heavy bootstrap now allows up to three minutes to finish because large pools can take longer than 60s on the first run.

### 4. Updated Frontend
**File:** `app/page.tsx` (line 228-270)
- `fetchDynamicMetrics()` calls `ApiClient.computeStrategy()` and always sends the active pool + factor selections so the lightweight path honors single or multi-factor combos.

### 5. Updated API Client
**File:** `lib/api-client.ts`
- Extended `StrategyRequest` typing with `pool_id`, `factor_ids`, and cached signal path.

### 6. Added skip_analytics to compute_service.py
**File:** `python-backend/compute_service.py` (line 266, 587)
- Added `skip_analytics` parameter to skip VIF, grid search, rolling IC
- VIF/correlation grids are only produced during the heavy compute path when explicitly requested; the lightweight strategy endpoint never recomputes them.

---

## Suspected Problems

### 1. Frontend Still Calling Old Endpoint?
Check if `fetchDynamicMetrics()` is actually being called, or if another function like `fetchFactorDataset()` is being triggered instead.

### 2. Python Path Resolution
The route might not find the correct Python or script path:
```typescript
// In route.ts
const venvPython = path.resolve(process.cwd(), '../.venv/bin/python');
const scriptPath = path.resolve(process.cwd(), '../python-backend/strategy_service.py');
```
If `process.cwd()` is wrong, it may fall back to system python3 without required packages.

### 3. Binary File Reading Issue
`strategy_service.py` reads from:
```python
qlib_path = str(PROJECT_ROOT / "data" / "1555_qlib")
```
If `PROJECT_ROOT` resolves incorrectly, file loading fails.

### 4. Next.js Caching
Next.js might be caching the old API response or route.

### 5. Horizon Mismatch (Charts Empty)
The dashboard filters `daily_returns` by `horizon_days === target_horizons[0]`. If the strategy backend hardcodes `horizon_days` (e.g. always `1`), the API can return data but the charts will be empty after filtering.

Fix: include `target_horizons` in the `/api/v1/strategy` request and have `strategy_service.py` label `metrics[].horizon_days` and `daily_returns[].horizon_days` with that value. (Implemented.)

### 6. Identical Charts Across Factors/Selections
If `/api/v1/strategy` is called without `factor_signals`, `strategy_service.py` falls back to a constant signal and the backtest becomes independent of:
- selected factors / factor combinations
- pool selection

Symptom: every backtest produces the same equity curve and the same performance ticker.

Fix: the strategy API now materializes/caches a combined factor signal per `(pool, factor_ids, dates)` by calling `compute_service.py` once with `skip_analytics:true`. Subsequent lightweight strategy runs reuse that `.npy`, so single/multi-factor selections map 1:1 to the custom curve the UI expects.

---

## Files Involved

### Python Backend
| File | Size | Purpose |
|------|------|---------|
| `python-backend/compute_service.py` | 32KB | Full compute (heavy) |
| `python-backend/strategy_service.py` | 6KB | Lightweight strategy |
| `python-backend/requirements.txt` | 106B | Dependencies |

### API Routes
| File | Endpoint |
|------|----------|
| `app/api/v1/compute/metrics/route.ts` | POST /api/v1/compute/metrics |
| `app/api/v1/strategy/route.ts` | POST /api/v1/strategy |

### Frontend
| File | Function |
|------|----------|
| `app/page.tsx` | `fetchDynamicMetrics()`, `fetchFactorDataset()` |
| `lib/api-client.ts` | `ApiClient.computeMetrics()`, `ApiClient.computeStrategy()` |

### Data
| Path | Size | Content |
|------|------|---------|
| `data/1555_qlib/` | 17MB | Qlib binary data |
| `data/1555_qlib/features/*/close.day.bin` | ~4KB each | Close prices per stock |
| `data/1555_qlib/calendars/day.txt` | 16KB | Trading dates |
| `data/1555_qlib/instruments/all.txt` | 1KB | Stock list |

---

## Git Commands Used

### Initial Push (Fresh Repo)
```bash
cd /Users/ikki/AlphaSAGE/alpha-sage-dashboard

# Remove old git (was in parent AlphaSAGE/)
rm -rf .git

# Initialize fresh
git init
git branch -m main
git remote add origin https://github.com/Ikkisovi/alpha-dashboard.git

# Stage and commit
git add .
git commit -m "Initial commit: AlphaSAGE Dashboard with backend and data"

# Force push (replace broken history)
git push -f origin main
```

### Subsequent Commits
```bash
git add .
git commit -m "Commit message"
git push origin main
```

---

## Debug Steps to Try

### 1. Test strategy_service.py Directly
```bash
cd /Users/ikki/AlphaSAGE/alpha-sage-dashboard
echo '{"train_start":"2024-01-01","test_end":"2024-12-31","strategy":{"type":"long_short","top_pct":0.2,"rebalance_days":5}}' | python3 python-backend/strategy_service.py
```

### 2. Check API Route is Being Hit
Add logging to `app/api/v1/strategy/route.ts`:
```typescript
console.log('[Strategy API] Request received:', body);
```

### 3. Check Frontend is Calling Correct Endpoint
In browser DevTools Network tab, look for:
- `/api/v1/strategy` (lightweight) vs `/api/v1/compute/metrics` (heavy)

### 4. Check Python Path Resolution
```bash
# In dashboard directory
node -e "console.log(require('path').resolve(process.cwd(), 'python-backend/strategy_service.py'))"
```

### 5. Test Binary File Reading
```bash
python3 -c "
from pathlib import Path
p = Path('/Users/ikki/AlphaSAGE/alpha-sage-dashboard/data/1555_qlib/features/spy/close.day.bin')
print(f'Exists: {p.exists()}')
import numpy as np
data = np.fromfile(p, dtype='<f')
print(f'Shape: {data.shape}, Sample: {data[:5]}')
"
```

---

## Test Results

### strategy_service.py CLI Test: SUCCESS
```bash
echo '{"train_start":"2024-01-01",...}' | python3 python-backend/strategy_service.py
# Returns instantly with 246 daily returns
```

**Conclusion:** Python service works. Issue is in frontend/Next.js routing.

---

## Most Likely Cause

**Frontend is calling `fetchFactorDataset()` instead of `fetchDynamicMetrics()`**

Check in browser DevTools Network tab:
- If calling `/api/v1/compute/metrics` → wrong endpoint
- If calling `/api/v1/strategy` → correct endpoint

**Or:** Next.js dev server needs restart to pick up new route.

---

## Next Steps

1. **Restart Next.js dev server** (`npm run dev`)
2. **Check Network tab** - which endpoint is being called?
3. **Add console.log to page.tsx** to verify which function triggers
