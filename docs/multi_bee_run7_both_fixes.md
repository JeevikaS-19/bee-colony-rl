# Run 7: both fixes together (`mb_fresh_s0`, `mb_fresh_s1`)

Written 2026-11-02. Sources: `kaggle-checkpoints/multibee_checkpoints/mb_fresh_s{0,1}/`, `kaggle-checkpoints/logs/`.
Setup: `--bees 1 --steps 30000 --read-start 999999 --buffer 200000 --chunk-steps 5 --max-lag 10 --sync-every 5 --warmup-steer 0.3 --warmup-throttle-min 0.2`, shaping off, seeds 0 and 1.

| | s0 | s1 |
|---|---|---|
| mean lag (updates) | 17.3 (max 20) | 17.9 (max 19) |
| warmup path mean/median | 40.8 / 36.6 m | 35.4 / 29.4 m |
| stall ends | crashes from 7.5-10k | 12.5-15k |
| first arrival | step 24,524 (ep 142) | none |
| 20-30k | 1 arrival, 42 crashes/timeouts | 100% timeouts again (relapse), max path 36.8 m |

Single-bee references: exit ~14.5k / first arrival 27.7k (s0); exit 9.4k (s1).

## Reading
- Neither fix alone worked (`mb_warm_s1`, `mb_tight_s1`); together both seeds leave the stall. Seed 0 matches or beats single-bee; seed 1 exits later and relapses after 20k (cause unknown).
- One seed per cell. Only `--bees 1`, no shaping, no reading: says nothing yet about multiple bees or pheromones.

## Next
3-bee control (shaping off) vs 3-bee shaping on, same settings, seeds 0 and 1, 60k steps. See STATUS.md.
Note: with option A, NOVICE arrivals write NO success trace and promotion to FORAGER needs E>=0.3 (about 4 arrivals), so one arrival at 24.5k leaves the success channel empty.

## Correction and code change (same day)
- Correction to my earlier claim: the v2 table the loop uses DOES store dances with `heading` and `steer` (not throttle), and `shaping_calculator.py` rewards heading along nearby success dances (`cos(agent_heading - dance_heading)`, weight = spatial falloff x decayed vigor, scaled by `eta_success` 0.5). Still reward-side, not action-side.
- Added `--novice-writes` (default off = option A, so earlier runs reproduce). With it, NOVICE arrivals write success dances at vigor 0.05. Recorded in `run_config.json` as `novice_writes`.
- Caveat: at vigor 0.05 the per-step shaping is at most about 0.5 x 0.05 = 0.025 and the trace falls below `min_vigor` (0.01) after about 2.3 half-lives (about 20k steps at half-life-frac 0.15 over 60k). So the signal may be too weak to move anything. A null result then means "too faint", not "pheromones useless". Disclose either way.
- Zip: `bee_multi_v7.zip` (13 files, imports checked from an empty folder).
