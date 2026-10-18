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

- `β > 0` is the scale. `λ ≥ 0` weights warnings against successes. Start with `β` chosen so that a full-strength success trace gives a shaping bonus of about 10% of a typical arrival reward, and `λ = 1`. Both are free parameters to report, not tuned on the test.
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

- **(A) Novice warnings at low vigor.** NOVICE bees write warnings at a fixed vigor below the FORAGER minimum (0.1), and never write successes. `E_t` stays terminal-only. This changes the NOVICE row of the caste table.
- **(B) Partial credit.** `E_t` gains a small amount per unit of route progress. This breaks the "never credited per step" rule in CLAUDE.md, which was a deliberate decision.
- **(C) Keep both rules.** Accept that guidance cannot affect time-to-first-arrival. Report the time-to-first-arrival of each arm as a separate result.

Recommendation: (A) for the first multi-bee run, because it keeps the terminal-only credit rule and gives novices a warning signal from their crashes. Confirm before wiring the tracker.

## 9. Parameters to report

`β`, `λ`, σ = 6 m, sensor radius = 10 m, half-life (fraction of steps), snapshot interval K, γ = 0.99, horizon 500, penalties 20.0, warmup 5000 with throttle bias, `target_entropy = 0.0`, batch 256, learning rate 3e-4, seed. All identical across arms except the shaping switch.
