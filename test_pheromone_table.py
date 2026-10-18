"""
test_pheromone_table.py

Test harness to rigorously exercise and validate the PheromoneTable class.
Verifies grid isolation, max-vigor gating, spatial falloff, empty reads, and decay.
"""

import sys
import os
import math
import unittest.mock as mock

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pheromone_table import PheromoneTable, decay_rate_from_half_life

def run_tests():
    print("=== Testing Step 2: Stigmergic Pheromone Table Validation ===\n")

    table = PheromoneTable(grid_size=2.0)

    coord_center = (15.0, -23.0)
    expected_key = (7, -12)

    print(f"Test Coordinate: {coord_center} -> Expected Grid Cell Key: {expected_key}")
    assert table._to_grid_key(coord_center) == expected_key, "FAIL: Grid coordinate conversion mismatch."
    print("-> SUCCESS: Grid coordinate bucketing works flawlessly!\n")

    #Test 1: WRITE + READ AT THE SAME PLACE
    print("Test 1: Basic Write + Read (Immediate Feedback Loop)")

    # Write a success trace with 0.5 vigor
    meta_1 = {"worker_id": "Forager_7", "trigger": "trajectory_completion"}
    write_success = table.write(coord_center, "success", vigor=0.5, step=0, meta=meta_1)
    assert write_success is True, "FAIL: Initial write failed."

    # Same step as the write -> no decay. decay_rate=0.0 isolates spatial scaling.
    sensed = table.read_nearby(coord_center, step=0, sensor_radius=10.0, sigma=6.0, decay_rate=0.0)
    print(f"  Sensed Success Intensity: {sensed['success']:.4f} (Original Vigor: 0.5)")

    # Since we are reading from the center of the cell, the spatial weight is exactly 1.0
    assert abs(sensed["success"] - 0.5) < 0.001, f"FAIL: Expected 0.5, got {sensed['success']}"
    print("-> SUCCESS: Basic write/read verified with zero-distance spatial matching!\n")

    # Test 2: Overwrite-if-higher-vigor (Both Directions)
    print("--- Test 2: Overwrite-if-higher-vigor (Both Directions) ---")

    new_coord_center = (41.0, 51.0)

    # 1. Write lower vigor (0.3)
    res_3 = table.write(new_coord_center, "success", vigor=0.3, step=0, meta={"worker_id": "Bee_A", "trigger": "straight_path"})
    assert res_3 is True
    assert table.table[table._to_grid_key(new_coord_center)]["success"]["vigor"] == 0.3

    # 2. Try to overwrite with HIGHER vigor (0.6) -> MUST succeed
    res_6 = table.write(new_coord_center, "success", vigor=0.6, step=0, meta={"worker_id": "Bee_B", "trigger": "smooth_turn"})
    assert res_6 is True, "FAIL: Higher vigor write should have updated cell."
    assert table.table[table._to_grid_key(new_coord_center)]["success"]["vigor"] == 0.6, "FAIL: Vigor value did not update to 0.6."
    print("  -> Accept Higher Vigor (0.3 -> 0.6): Passed!")

    # 3. Try to overwrite with LOWER vigor (0.2) -> MUST fail & return False
    res_2 = table.write(new_coord_center, "success", vigor=0.2, step=0, meta={"worker_id": "Bee_C", "trigger": "jittery_driving"})
    assert res_2 is False, "FAIL: Lower vigor write should have returned False (rejected)."
    assert table.table[table._to_grid_key(new_coord_center)]["success"]["vigor"] == 0.6, "FAIL: Lower vigor hijacked cell!"
    print("  -> Reject Lower Vigor (0.6 -> 0.2): Passed!")
    print("-> SUCCESS: Overwrite-if-higher-vigor enforces spatial data trust boundaries!\n")

    # Test 3: Two Independent Channels are Independent
    print("Test 3: Decoupled Success/Warning Channels")

    chan_coord_center = (-9.0, 81.0)

    # Write a warning at the spot (a vehicle crashed here)
    meta_crash = {"worker_id": "Novice_9", "trigger": "curb_crash"}
    write_warn = table.write(chan_coord_center, "warning", vigor=0.8, step=0, meta=meta_crash)
    assert write_warn is True

    # Write a success at the exact same spot (a veteran squeezed past safely)
    meta_safe = {"worker_id": "Scout_0", "trigger": "precision_pass"}
    write_succ2 = table.write(chan_coord_center, "success", vigor=0.9, step=0, meta=meta_safe)
    assert write_succ2 is True

    # Read both channels at this coordinate (decay = 0 to isolate)
    sensed_chan = table.read_nearby(chan_coord_center, step=0, sensor_radius=5.0, decay_rate=0.0)
    print(f"  Sensed at collision spot:")
    print(f"    Success Channel: {sensed_chan['success']:.4f} (Expected: 0.9000)")
    print(f"    Warning Channel: {sensed_chan['warning']:.4f} (Expected: 0.8000)")

    assert abs(sensed_chan["success"] - 0.9) < 0.001, "FAIL: Success channel corrupted or missing."
    assert abs(sensed_chan["warning"] - 0.8) < 0.001, "FAIL: Warning channel corrupted or missing."
    print("-> SUCCESS: Double-channel isolation preserves danger warnings beside safe paths!\n")

    # Test 4: Spatial Falloff Works
    print("Test 4: Spatial Falloff (Sensing Range Footprints)")

    # Center of cell key (50, 50) -> center: (101.0, 101.0)
    source_coord_center = (101.0, 101.0)

    # Write a perfect trace (vigor 0.95)
    table.write(source_coord_center, "success", vigor=0.95, step=0, meta={"worker_id": "Scout_1", "trigger": "test_falloff"})

    #test reading with sensor_radius = 10.0 meters and sigma = 6.0

    # A. Just inside the radius (8.0 meters away)
    # Distance from (109.0, 101.0) to (101.0, 101.0) is exactly 8.0 meters.
    # 8.0 meters is < 10.0 (inside radius).
    # Spatial weight is exp(-(8.0^2)/(2*6.0^2)) = exp(-64 / 72) = exp(-0.88889) ≈ 0.41111
    # Expected reading: 0.95 * 0.41111 ≈ 0.3905
    sensed_inside = table.read_nearby((109.0, 101.0), step=0, sensor_radius=10.0, sigma=6.0, decay_rate=0.0)
    print(f"  Reading just inside radius (8.0m away): {sensed_inside['success']:.4f} (Expected: ~0.3906)")
    assert sensed_inside["success"] > 0.0, "FAIL: Sensed zero inside the sensor radius."
    assert abs(sensed_inside["success"] - 0.3906) < 0.005, f"FAIL: Expected spatial discount weight mismatch. Got {sensed_inside['success']}"

    # B. Just outside the radius (11.0 meters away)
    # Distance from (112.0, 101.0) to (101.0, 101.0) is exactly 11.0 meters.
    # 11.0 meters is > 10.0 (outside radius). Must return 0.0.
    sensed_outside = table.read_nearby((112.0, 101.0), step=0, sensor_radius=10.0, sigma=6.0, decay_rate=0.0)
    print(f"  Reading just outside radius (11.0m away): {sensed_outside['success']:.4f} (Expected: 0.0000)")
    assert sensed_outside["success"] == 0.0, "FAIL: Sensed pheromone leaks outside the physical sensor footprint!"
    print("-> SUCCESS: Spatial Gaussian falloff and distance constraints verified!\n")

    # Test 5: Empty Read (Uncharted Territory)
    print("--- Test 5: Empty Read (Uncharted Territory) ---")

    empty_coord = (-500.0, -500.0)

    #Read sitting in clean territory
    sensed_empty = table.read_nearby(empty_coord, step=0, sensor_radius=10.0, decay_rate=0.0)
    print(f"  Sensed in clean air: Success: {sensed_empty['success']:.1f} | Warning: {sensed_empty['warning']:.1f}")
    assert sensed_empty["success"] == 0.0
    assert sensed_empty["warning"] == 0.0
    print("-> SUCCESS: Empty reads gracefully return baseline zero matrices!\n")

    # Test 6: Decay counts training steps, not wall-clock time
    print("Test 6: Step-based temporal decay")
    decay_table = PheromoneTable(grid_size=2.0)
    # (5.0, 5.0) is the exact centre of cell (2, 2), so spatial weight is 1.0
    decay_table.write((5.0, 5.0), "success", vigor=1.0, step=100)

    # 100 steps later at 0.01/step -> exp(-1)
    later = decay_table.read_nearby((5.0, 5.0), step=200, sensor_radius=10.0, decay_rate=0.01)
    assert abs(later["success"] - math.exp(-1.0)) < 1e-6, f"FAIL: expected exp(-1), got {later['success']}"

    # Same step as the write -> no decay
    same = decay_table.read_nearby((5.0, 5.0), step=100, sensor_radius=10.0, decay_rate=0.01)
    assert abs(same["success"] - 1.0) < 1e-6, f"FAIL: expected 1.0 at write step, got {same['success']}"

    # Reading at an earlier step (delta < 0) clamps to no decay, never amplifies
    earlier = decay_table.read_nearby((5.0, 5.0), step=50, sensor_radius=10.0, decay_rate=0.01)
    assert abs(earlier["success"] - 1.0) < 1e-6, f"FAIL: expected 1.0 for read before write, got {earlier['success']}"
    print(f"  100 steps at 0.01/step: {later['success']:.4f} (expected {math.exp(-1.0):.4f})")
    print("-> SUCCESS: Decay is driven by training steps, independent of wall-clock time!\n")

    # Test 7: zero-vigor writes are rejected, not silently stored
    print("Test 7: Zero-vigor write is rejected")
    zero_table = PheromoneTable(grid_size=2.0)
    try:
        zero_table.write((1.0, 1.0), "success", vigor=0.0, step=0)
        raise AssertionError("FAIL: vigor=0.0 write was accepted")
    except ValueError:
        pass
    assert zero_table.table == {}, "FAIL: rejected write still created a cell"
    print("-> SUCCESS: vigor <= 0 raises and leaves the table untouched!\n")

    # Test 8: half-life -> per-step rate. Half-life of 100 steps must halve a trace at step 100.
    print("Test 8: Half-life to per-step decay rate")
    rate = decay_rate_from_half_life(100)
    assert abs(rate - math.log(2) / 100) < 1e-12, f"FAIL: rate {rate}"
    assert decay_rate_from_half_life(None) == 0.0, "FAIL: None should mean no decay"
    assert decay_rate_from_half_life(0) == 0.0, "FAIL: 0 should mean no decay"
    half_table = PheromoneTable(grid_size=2.0)
    half_table.write((5.0, 5.0), "success", vigor=1.0, step=0)
    halved = half_table.read_nearby((5.0, 5.0), step=100, decay_rate=rate, sensor_radius=10.0)
    assert abs(halved["success"] - 0.5) < 1e-6, f"FAIL: expected 0.5 after one half-life, got {halved['success']}"
    print(f"-> SUCCESS: half-life 100 steps halves the trace to {halved['success']:.4f}!\n")

    print("CONGRATULATIONS: All 8 core test cases passed with zero failures!")



if __name__ == "__main__":
    run_tests()
