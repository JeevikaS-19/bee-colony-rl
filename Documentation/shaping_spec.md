# Reward shaping from pheromone traces: specification

Status: **draft, not implemented.** Written before the code, per the review. Every equation here is the definition the code must match. Change this file first if the design changes.

## 1. Setting

- One bee, MetaDrive map C, one fixed scenario (`Scenario Index 0`), horizon H = 500.
- Environment reward `r_t` is MetaDrive's own (progress term plus terminal penalties of 20.0 for out-of-road and crashes).
- The pheromone table `P` holds, per 2 m grid cell, two channels: `success` and `warning`. Each stores vigor, the training step it was written at, and the speed and steering of the bee when it passed (see §6).
- Reads are `S(x) = read_nearby(x)["success"]` and `W(x) = read_nearby(x)["warning"]`. Both are in [0, 1], with Gaussian spatial falloff (σ = 6 m, radius 10 m) and exponential temporal decay (per training step, half-life set by `--half-life-frac`).

## 2. Potential

A potential is a function of the state. Here the state is the bee's position `x_t` plus the table `P`:

  Φ_t(s) = β · ( S(x) − λ · W(x) )

- `β > 0` is the scale. `λ ≥ 0` weights warnings against successes. **Frozen: β = 1.0, λ = 1.0.** β is set on the per-step scale, not the episode scale: the logged environment return is about 1 per step (e.g. 240 over ~200 steps), so β = 1 makes |F| comparable to the environment reward. (An earlier draft proposed "10% of an arrival reward", which does not give a usable number on the per-step scale, so it is dropped.) Both are reported, not tuned on the test.
- Terminal states have potential 0: `Φ(s_T) = 0` for `s_T` a real terminal (arrival, crash, out-of-road).
- Timeouts (horizon truncation) are not absorbing. Use the actual `Φ` at the truncated state.

## 3. Shaping term and shaped reward

Potential-based shaping (Ng, Harada & Russell, 1999, "Policy invariance under reward transformations"; *verify citation and page before the manuscript*):

  F(s_t, s_{t+1}) = γ · Φ(s_{t+1}) − Φ(s_t)

Shaped reward used for the update:

  r'_t = r_t + F(s_t, s_{t+1})

For a real terminal, `Φ(s_{t+1}) = 0`, so `F = −Φ(s_t)`.

## 4. Why this form, and the non-stationarity problem

**Static Φ.** If `Φ` is fixed, the shaping is a telescoping sum. Over any closed loop of states the total `Σ γ^t F` is zero. A bee that circles a success trace collects no net bonus. In that case the optimal policy is unchanged by the shaping. This is the "doesn't hack the shaping" guarantee, and it holds only for a fixed `Φ`.

**Live Φ.** The table changes as the bee writes traces. Then `Φ_t` depends on `t`, and the telescoping argument fails. A bee could be rewarded for returning to a cell it has just reinforced. This breaks the invariance guarantee.

**Proposed fix (recommended, to confirm):** freeze `Φ` in snapshots. The table is read-only during an interval of K episodes. At the interval boundary it is updated with the new traces and frozen again. Within each interval the static-Φ guarantee holds. Report K and the freeze rule with the results.

**What this means for the arms (a consequence to report).** Over an episode that ends at a real terminal, `Φ(s_T) = 0`, so the total shaping is `Σ γ^t F_t = −Φ(s_0)`. That is the same for every path from the same spawn. The shaping therefore does not change which path has the higher *undiscounted total*. Any effect of shaping on learning comes through the per-step credit signal and the discounting, not through the episode total. Timeouts (truncations) are different: their total is `γ^T Φ(s_T) − Φ(s_0)`, which depends on the endpoint. The single-bee arms should be read with this in mind.

## 5. Checks the single-bee test must pass

The single-bee self-reading test (§7) is there to check the shaping, not to make the claim. It passes only if all four hold:

1. **No freeze.** Mean forward distance per episode stays above a floor (e.g. > 20 m after warmup) for the shaped arm. The unshaped arm is the reference.
2. **No hacking.** Shaped-minus-unshaped environment return does not rise while arrival rate and the progress term stay flat. Check for loops: count repeated cells within an episode.
3. **Still learns.** Arrival rate or mean environment reward at the same step count is at least as good as the unshaped arm.
4. **Shaping is bounded.** Mean |F| per step is a small fraction of mean |r| per step. Report both.

## 6. What a trace carries (decided)

Each trace stores, in addition to vigor, timestamp, and meta:

- `speed` (km/h) at the cell, when the bee passed it, and
- `steer` (commanded steering, −1..1).

Reason: the crash spot in the 40k h500 run has a success trace on the same 2 m cell as its warning. Location alone cannot say how to pass. Speed and steer can. Vigor still decides which trace survives in a cell; speed and steer are replaced together with vigor.

Not used by Φ yet. Using them (e.g. a success bonus only when `|speed − v_trace| < δ`) is a later spec change, written here before any code.

## 7. The single-bee self-reading test

Purpose: check §5. It is not the research claim.

- One bee. It reads only its own traces. Other bees do not exist.
- Only warnings are guaranteed by its own experience from episode 1: a single bee crashes early, so `W` is nonzero early. Successes need a first arrival, so `S` is zero until the first arrival. This is the cold-start issue of §8, but for one bee the warning channel is available at once.
- Arms, run with identical seed, warmup, horizon, penalties, and hyperparameters: (a) no shaping, (b) shaping with a static Φ snapshot, (c) shaping with live Φ. Arm (c) is expected to show the hacking risk of §4.

## 8. Cold start (open decision)

`E_t` starts at 0 and rises only on terminal arrivals, which is the existing rule (CLAUDE.md). NOVICE bees have vigor 0, so they write nothing. Until the first arrival of any bee there are no success traces. Early on, every bee is a NOVICE, so the shaping term `Φ` is driven only by warnings.

Options:

- **(A) Novice warnings at low vigor. APPROVED for the NOVICE row.** NOVICE bees write warnings at a fixed vigor of **0.05** (clearly below the FORAGER floor of 0.1, so the hierarchy stays visible), and never write successes. `E_t` stays terminal-only. This changes the NOVICE row of the caste table. Code not yet changed.
- **(B) Partial credit.** `E_t` gains a small amount per unit of route progress. This breaks the "never credited per step" rule in CLAUDE.md, which was a deliberate decision.
- **(C) Keep both rules.** Accept that guidance cannot affect time-to-first-arrival. Report the time-to-first-arrival of each arm as a separate result.

Recommendation: (A) for the first multi-bee run, because it keeps the terminal-only credit rule and gives novices a warning signal from their crashes.

### 8a. Replay of the h500 outcomes through `ReputationTracker` (done, offline)

Input: the 200 episode outcomes in `logs/train_100k_h500_seed0.log` (10 arrive, 164 crash, 26 timeout). Timeouts give no terminal credit. Current steps are +0.05 / −0.10 with a floor of 0.

| Step sizes (up / down) | Final E | Max E | First FORAGER (≥0.3) | First SCOUT (≥0.8) |
|---|---|---|---|---|
| +0.05 / −0.10 (current) | 0.000 | 0.150 | never | never |
| +0.05 / −0.05 | 0.000 | 0.150 | never | never |
| +0.05 / −0.02 | 0.000 | 0.200 | never | never |
| +0.05 / −0.0125 | 0.075 | 0.275 | never | never |
| +0.10 / −0.10 | — | 0.300 | reached | never |
| +0.10 / −0.05 | — | 0.300 | reached | never |
| +0.10 / −0.025 | — | 0.550 | reached | reached |
| +0.20 / −0.05 | — | 1.000 | reached | reached |

Facts from the same log:

- Overall arrival rate: **5%** (10/200). The best 20-episode window: **40%** (ends at episode 184). The last 20 episodes: 15%.
- Longest run of consecutive arrivals: **3**. Arrivals are never in long runs, so "six in a row" is not the constraint; the binding constraint is the overall rate.
- The whole run's arrivals fall in episodes 148–200. Before episode 148, E = 0 throughout.

Conclusion: under this policy's outcome mix, the current steps and thresholds never promote a bee. The step sizes must be set against an outcome rate, not against an ideal streak. The thresholds (0.3 / 0.8) and the step sizes have to be fixed before the comparison runs and must not be fitted to the baseline's own outcomes, or the comparison is circular.

Options for the step rule (not yet chosen): (i) keep the steps, lower the FORAGER threshold to match achievable E; (ii) raise up and lower down together so the ratio sets the drift; (iii) an exponentially weighted arrival rate, where E tracks the arrival fraction directly and FORAGER means a fixed arrival rate. Option (iii) keeps the thresholds meaningful, but a 0.3 threshold means a 30% arrival rate, which this policy reaches only in a short window.

Plot: `scratchpad\reputation_replay_h500.png` (outside the repo).

### 8b. Decision: option 3, exponentially weighted arrival rate

- `E <- (1 - alpha) E + alpha * outcome`, outcome 1 for arrival and 0 for crash, terminal outcomes only. Timeouts do not update.
- Castes unchanged: FORAGER at 0.3 (about a 30% recent arrival rate), SCOUT at 0.8.
- alpha = **0.1**. Caveat: 0.05, 0.1, and 0.2 were all replayed on the h500 outcomes before 0.1 was chosen. Replay at 0.1: first FORAGER at episode 181, 6 episodes in FORAGER, no SCOUT. Before the comparison runs, alpha must be fixed in this spec and not changed afterward.
- NOVICE vigor 0.05 (approved). NOVICE arrivals write no success trace. NOVICE crashes still write warnings.
- Wired into the single-bee loop (`REPUTATION_ALPHA`). Multi-bee and the uniform-weight arm are not built.

## 10. Frozen settings for the single-bee arms

Frozen before any comparison run. Changing one means the runs are redone.

| Setting | Value |
|---|---|
| Reputation smoothing (EWMA weight) | 0.1 (`REPUTATION_SMOOTHING`) |
| Reputation outcome | arrive = 1; crash, out-of-road, and horizon timeout = 0 |
| Caste thresholds | FORAGER ≥ 0.3, SCOUT ≥ 0.8 |
| Hysteresis | 0.05 (promote at the threshold, demote 0.05 below it) |
| NOVICE vigor | 0.05 |
| NOVICE success writes | **none** (option A). Reviewer proposed allowing faint novice success writes at 0.05. Not adopted; open question, see the checklist |
| Shaping β, λ | 1.0, 1.0 |
| Shaping arms | none; snapshot with K = 10 episodes; live (K = 1) |
| γ | 0.99 |
| Pheromone decay | half-life 0.15 × total steps |
| Sensor radius, σ | 10 m, 6 m |
| Horizon, penalties | 500; 20.0 for out-of-road, crash_vehicle, crash_object |
| Warmup, batch, lr, target entropy | 5000 steps (throttle-biased); 256; 3e-4; 0.0 |
| Seed | 0 (one seed per arm; not yet multi-seed) |

**Pilot finding (stop condition hit, snapshot and live arms NOT run).** Mean |F| per step on the 2000-step snapshot smoke run was 0.00014, against mean |raw reward| of 0.099 per step: a ratio of about 0.14%, roughly 700 times smaller. Cause: with no FORAGER bee there are no success traces, so Φ = −β·λ·W with W at most 0.05 (novice vigor). So |Φ| ≤ β·λ·0.05 per step, and mean |F| scales linearly with β. Under these settings the snapshot and live arms would reproduce the none arm. The none arm was left running as the reference. Decision needed on scale before any shaped arm is run: β (and/or λ), the novice floor, or whether novices may write success traces (the open question above). Rough sizing from the smoke run: about β = 70 would bring mean |F| to 10% of mean |raw|. This estimate is from 2000 steps and should be redone on the none arm's trace data.

### 10a. Beta rule (written before any shaped arm runs)

**Rule.** β is set so that mean |F| per step equals 10% of mean |r| per step, measured on the **final checkpoint of the `shaping_none_s0` arm** with every trace's vigor rescaled to 1.0. Measured by `measure_beta.py` (10 rollouts, stochastic policy, seed 0; decay off; λ = 1). Because F is linear in β, one rollout at β = 1 gives β = 0.10 × mean|r| / mean|F_β=1|.

**The 10% is a judgment call.** It is not derived from a theory. It sets the shaping at a small but visible fraction of the per-step reward.

**Why vigor 1.0.** A single bee is NOVICE for most of a run by design, so vigor-weighted traces make the guidance tiny by construction. Rescaling isolates the mechanism's scale from the reputation mechanism, which the multi-worker stage will test.

**Result (frozen):**

| Quantity | Value |
|---|---|
| Checkpoint | `shaping_none_s0`, final (60k steps) |
| Rollouts | 10 (4,280 steps) |
| mean \|r\| per step | 0.5377 |
| mean \|F\| per step at β = 1 | 0.0188 |
| Fraction of steps with Φ ≠ 0 | 0.940 |
| Cells in table (rescaled) | 345 |
| **β** | **2.857 → set β = 2.86, λ = 1.0** |

Sensitivity note: an earlier measurement on the 20k checkpoint (53 cells, 5 rollouts) gave β = 1.62. The value depends on the checkpoint, so the rule fixes it at the final checkpoint. Report both.

**Pilot watch-list.** The positive potential on success traces is about 2.86 per cell at full vigor. Approaching a region with a positive potential is rewarded. Entering a warning region is penalized. A "hill" just before the failure zone could bring back the freeze, because approaching it is penalized. Check distance and max speed in the pilot logs, not reward alone.

### 10b. Pilot results (seed 0, 60k steps, stub vigor 1.0, beta 2.86)

- **Plumbing: pass.** All 71 episodes that end before the read gate (step 5000) match exactly across `none`, `pilot_snap10_s0`, and `pilot_live_s0`. After the gate they diverge, as intended.
- **Scale: pass.** Mean |F| / mean |raw| per step = 0.13 (snapshot) and 0.11 (live). The target was 0.10.
- **Freeze: FAIL in both pilots.** Episodes 121-200 of the snapshot arm and 121-160 of the live arm end almost entirely by out-of-road within a few metres of spawn (median 3.5 m and 4.2 m). The `none` arm reaches a median of 155 m in its 161-200 window. Single seed, so this is not attributable yet.
- **Mechanism: not identified.** The spawn has success and warning traces within 2 m of each other, so Phi there is near zero. The hypothesis that a negative potential at spawn makes leaving the road pay was checked and does not hold for this table. Candidate causes, not tested: (a) the terminal convention (Phi = 0 at a real terminal) paying out F = -Phi(s) when a crash ends the episode from a warned cell; (b) overlapping success and warning traces at vigor 1.0; (c) seed noise.
- **Hacking: not measured on the `none` arm.** It has no shaping log, so the distinct-cell fraction (0.16-0.18 in the pilots) has no baseline yet.

**Next (not yet run):** a second seed for both pilots; read the first steps of the collapsed episodes (steering and throttle); and a control with the terminal convention changed. No arm counts are claimed from the pilot.

### 10c. Second-seed pilot (seed 1, same settings)

- **Plumbing: pass again.** 74 episodes before the gate match exactly across `none_s1`, `pilot_snap10_s1`, and `pilot_live_s1`.
- **Snapshot: no sustained freeze.** Episodes 121-200 recover to a median of 131-147 m, with max speed around 20 km/h. The seed-0 out-of-road collapse did not reproduce.
- **Live: sustained stall.** Episodes 81-200 have mean distance 9-28 m and max speed 5-8 km/h. Most episodes end in 500-step timeouts (32 in 121-160, 37 in 161-200). This is a freeze-like pattern, and it is a different failure from seed 0's out-of-road exits.
- **Baseline dip.** The `none` reference also slows around episodes 81-113 (mean 11 m, max speed 6 km/h), so part of the dip is not caused by shaping. The `none_s1` reference was still running at 113 episodes when this was written. Its episodes 121-200 are needed before the stall can be attributed.
- **Summary across 4 shaped runs:** a sustained slowdown appears in 3 of 4 (seed 0 snapshot and live, seed 1 live). Seed 1 snapshot does not show it. This is not enough to attribute the cause, and the arm comparison needs more seeds.

### 10d. Cheap checks on the seed-0 snapshot pilot (no training)

- **Phi map.** Spawn (5,7) has Phi = 0.16. Success traces give positive Phi on and west of the spawn (up to about 1.1 at x = -4). Warnings give negative Phi to the east and south (down to about -1.2 near (10..14, -4..-10)). Far from any trace Phi = 0. The table carries no road geometry, so "is leaving the road uphill" cannot be read from it. None of 8 rollouts left the road, so the exit question is untested.
- **Where |F| lives (8 rollouts, 3553 steps).** Median |F| 0.036, p99 0.25, mean 0.063. Mean |F| / mean |r| = 0.114. Steps within 10 m of spawn: 10% of steps but 1% of |F| mass.
- **Terminal dominance.** Seven arrival steps (0.2% of steps) carry 9% of total |F|, with F = -2.85 each. The arrival cell has a success trace (Phi about +2.85), and the terminal convention sets Phi(s_T) = 0. So the shaping removes the potential on the last step, even though the bee has arrived. Mean |F| looks small, but the tail is local.
- **Implication.** Local dominance sits at the destination, not at spawn. This is a candidate for the seed-0 collapse, but the collapse itself was not reproduced by these rollouts.

**Static-from-gate test started (seed 0, 60k steps, fixed vigor 1.0, beta 2.86).** The table is frozen at step 5000 and never refreshed. Run name `pilot_static_s0`. If it still collapses, the next candidates are the terminal convention and the ridge shape. Output: `logs/pilot_static_s0.log`.

### 10e. Valley and warnings checks (no training)

- **Are the spawn-area warnings junk from warmup?** No. Within 15 m of spawn the seed-0 snapshot pilot wrote 71 warnings during the policy phase and 20 during warmup. The near-spawn warnings are mostly the policy's own crashes. That suggests a feedback loop: early exits write warnings at spawn, and those warnings keep the start state in the valley.
- **Does forward descend into a valley?** Only slightly. Summing F over the first 15 m of six successful rollouts gives mean -0.06 (four negative, two positive, range -0.14 to +0.11). The raw reward over the same stretch is about 30. The valley exists but is too small to explain the collapse.
- **Steps-aligned comparison** (mean distance of episodes ending in each 10k-step window):

| Arm | 10k | 20k | 30k | 40k | 50k | 60k |
|---|---|---|---|---|---|---|
| none s0 | 35 | 12 | 69 | 153 | 138 | 207 |
| snapshot s0 | 35 | 9 | 16 | 46 | 161 | 201 |
| live s0 | 35 | 7 | 38 | 132 | 139 | 216 |
| none s1 | 32 | 11 | 14 | 131 | 133 | 109 |
| snapshot s1 | 32 | 12 | 9 | 101 | 104 | 128 |
| live s1 | 32 | 10 | 4 | 12 | 18 | 44 |

  Seed 0: both shaped arms lag the reference at 30–40k steps and catch up by 50k. Seed 1: the snapshot arm lags and recovers; the live arm stays stuck through 60k.
- **Teacher-table test** (proposed by review, not built): a fresh learner reads a frozen table from a competent run. Not yet implemented.

Mechanism note: the h500 replay showed the step-based rule never promoted (see §8a). The EWMA rule with smoothing 0.1 was chosen after viewing 0.05, 0.1, and 0.2 on the same outcomes. Disclose this in the manuscript.

## 9. Parameters to report

`β`, `λ`, σ = 6 m, sensor radius = 10 m, half-life (fraction of steps), snapshot interval K, γ = 0.99, horizon 500, penalties 20.0, warmup 5000 with throttle bias, `target_entropy = 0.0`, batch 256, learning rate 3e-4, seed. All identical across arms except the shaping switch.
