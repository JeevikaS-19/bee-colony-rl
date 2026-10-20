"""
Isolated check for PheromoneTableV2. No MetaDrive, no training loop.
Prints a readable summary; the OK/MISMATCH marks are the checks.
"""
from pheromone_table_v2 import PheromoneTableV2, decay_rate_from_half_life, decayed_vigor

HALF_LIFE = 100          # steps
RATE = decay_rate_from_half_life(HALF_LIFE)
READ_STEP = 1000

# One drive along y=1, heading 0 (so heading bin 0). One outcome per episode.
EPISODES = [
    # (episode, step, channel, crash_type, vigor)
    (0, 0,   "success", "none",           0.9),
    (1, 200, "crash",   "crash_vehicle",  0.8),
    (2, 400, "crash",   "out_of_road",    0.7),
    (3, 600, "crash",   "crash_object",   0.6),
    (4, 800, "crash",   "crash_building", 0.5),
]


def mark(ok):
    return "OK      " if ok else "MISMATCH"


def describe(dance, step_now):
    if dance is None:
        return "none"
    now = decayed_vigor(dance, step_now, RATE)
    return (f"crash_type={dance['crash_type']:<15} written@{dance['timestamp']:<4} "
            f"raw={dance['vigor']:.3f} now={now:.4f}")


def run_episodes():
    print("== 5-episode writes ==")
    table = PheromoneTableV2()
    for ep, step, channel, crash_type, vigor in EPISODES:
        x = 1.0 + 2.0 * ep  # each episode's outcome is one cell further along
        written = table.write_dance((x, 1.0), heading=0.0, channel=channel, steer=0.0,
                                    vigor=vigor, step=step, decay_rate=RATE, crash_type=crash_type)
        print(f"  ep{ep} step={step:<4} {channel}/{crash_type:<15} vigor={vigor} -> written={written}")

    print("\n== Cell contents at step", READ_STEP, "(decayed vigor shown as 'now') ==")
    checks = []
    for ep in range(len(EPISODES)):
        key = (ep, 0)  # x=1,3,5,7,9 -> grid x=0..4 with grid_size 2 (ep 0 -> key 0)
        s, cv, co = table.read_cell_dances(key, 0.0)
        print(f"cell {key}:")
        print(f"  success        : {describe(s, READ_STEP)}")
        print(f"  crash_vehicle  : {describe(cv, READ_STEP)}")
        print(f"  crash_offroad  : {describe(co, READ_STEP)}")

    # Expected: ep0 success; ep1 crash_vehicle; ep2-4 offroad with the right crash_type.
    s0, cv0, co0 = table.read_cell_dances((0, 0), 0.0)
    s1, cv1, co1 = table.read_cell_dances((1, 0), 0.0)
    _, cv2, co2 = table.read_cell_dances((2, 0), 0.0)
    _, _, co3 = table.read_cell_dances((3, 0), 0.0)
    _, _, co4 = table.read_cell_dances((4, 0), 0.0)
    print("\n== Routing checks ==")
    print(f"  {mark(s0 is not None and s0['crash_type'] == 'none')} ep0 success stored in success")
    print(f"  {mark(cv1 is not None and cv1['crash_type'] == 'crash_vehicle')} ep1 crash_vehicle in crash_vehicle channel")
    print(f"  {mark(co2 is not None and co2['crash_type'] == 'out_of_road')} ep2 out_of_road in offroad channel")
    print(f"  {mark(co3 is not None and co3['crash_type'] == 'crash_object')} ep3 crash_object rolled into offroad")
    print(f"  {mark(co4 is not None and co4['crash_type'] == 'crash_building')} ep4 crash_building rolled into offroad")


def run_decay():
    print("\n== Decay (half-life", HALF_LIFE, "steps) ==")
    dance = {"vigor": 0.8, "timestamp": 0}
    for age in (0, HALF_LIFE, 2 * HALF_LIFE):
        print(f"  age={age:<4} vigor {dance['vigor']:.3f} -> {decayed_vigor(dance, age, RATE):.4f}")
    half = decayed_vigor(dance, HALF_LIFE, RATE)
    print(f"  {mark(abs(half - 0.4) < 1e-9)} one half-life halves the vigor")


def run_overwrites():
    print("\n== Overwrite rule (new >= decayed old) ==")
    table = PheromoneTableV2()
    loc = (1.0, 1.0)
    w = lambda v, step: table.write_dance(loc, 0.0, "success", 0.0, v, step, RATE)
    print(f"  write 0.5 @0      -> {w(0.5, 0)}   (expect True, empty cell)")
    print(f"  write 0.9 @0      -> {w(0.9, 0)}   (expect True, bigger)")
    print(f"  write 0.3 @0      -> {w(0.3, 0)}   (expect False, smaller)")
    print(f"  write 0.3 @100    -> {w(0.3, 100)}   (stored 0.9 decays to 0.45, expect False)")
    print(f"  write 0.5 @100    -> {w(0.5, 100)}   (0.5 >= 0.45, expect True)")
    print(f"  write 0.0 @100    -> {w(0.0, 100)}   (vigor <= 0, expect False)")


def run_errors():
    print("\n== Rejected writes ==")
    table = PheromoneTableV2()
    for label, kwargs in [
        ("crash with crash_type none", dict(channel="crash", crash_type="none")),
        ("crash with unknown type", dict(channel="crash", crash_type="bogus")),
        ("unknown channel", dict(channel="warning", crash_type="crash_vehicle")),
    ]:
        try:
            table.write_dance((1.0, 1.0), 0.0, steer=0.0, vigor=0.5, step=0, decay_rate=RATE, **kwargs)
            print(f"  {mark(False)} {label}: no error raised")
        except ValueError as e:
            print(f"  {mark(True)} {label}: ValueError ({e})")
    try:
        decayed_vigor({"vigor": 0.5, "timestamp": 100}, 50, RATE)
        print(f"  {mark(False)} going back in time: no error raised")
    except ValueError as e:
        print(f"  {mark(True)} going back in time: ValueError ({e})")


if __name__ == "__main__":
    run_episodes()
    run_decay()
    run_overwrites()
    run_errors()
