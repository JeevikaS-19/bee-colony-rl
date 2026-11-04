# Run 8: 3 bees, no reading (control) vs reading + `--novice-writes` (treatment), 60k steps

Written 2026-11-03. Sources: `kaggle-checkpoints/checkpoints/mb3_ctrl_s0.log`, `mb3_ctrl_s1.log`, `mb_trt_s0.log` (the treatment seed 0 run; its run dir is `mb3_trt_s0`, the file name lacks the "3"; `mb3_trt_s0.log` is empty), `mb3_trt_s1.log`. Printed logs only, no CSVs or checkpoints were brought back.
Common flags: `--bees 3 --steps 60000 --chunk-steps 5 --max-lag 10 --sync-every 5 --warmup-steer 0.3 --warmup-throttle-min 0.2 --buffer 200000`. Control: `--read-start 999999`. Treatment: `--read-start 5000 --novice-writes`. Lag 22.6-23.2 mean, max 25 in all four.

| Run | First arrival | Arrivals | Last-window arrival rate (50-60k) | Final castes |
|---|---|---|---|---|
| control s0 | 27,096 | 73 | 0.89 | F, F, S |
| treatment s0 | 32,755 | 67 | 0.78 | F, F, F |
| control s1 | none | 0 | 0 (all timeouts) | N, N, N |
| treatment s1 | 39,210 | 47 | 0.97 | F, S, S |

## Observations
1. With both earlier fixes, 3 bees on a shared policy learn to arrive in 3 of 4 runs. The stall ends and arrivals rise from 0 to 0.8-0.97 per episode.
2. **No consistent pheromone effect.** Seed 0: treatment is later (32.8k vs 27.1k) with fewer arrivals. Seed 1: treatment arrives and control never does. One run each: this fits seed luck as well as an effect.
3. **Shaping is a tiny share of the return before the first arrival.** Mean F/r (shaping to reward ratio) after step 5k is 0.005-0.007 until about 30k, i.e. under 1%, and 0.046-0.070 only after arrivals are common. So the treatment cannot be what lets the policy escape the stall; it only becomes sizeable after arrivals exist.
4. Control seed 1 never leaves the timeout/crash regime in 60k (205 episodes, 0 arrivals), as seed 1 did in single-bee runs (relapse after 20k in `mb_fresh_s1`). Seed 1 seems harder or unlucky; n=1.
5. Not compared: a 1-bee run to 60k with the same flags, so "3 bees beat 1 bee" is not established.

## Not known
Seed noise size (need more seeds per arm); 1-bee 60k baseline; whether a stronger NOVICE vigor would change observation 3.
