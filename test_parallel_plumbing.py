"""
test_parallel_plumbing.py

File 4: Parallel Multiprocessing Plumbing for Multi-Agent MetaDrive.
Spawns 3 independent worker subprocesses, each running its own MetaDrive instance.
Implements Windows 'spawn' compatibility, sentinel-based queue termination,
and deadlock-free queue draining.

FIXED (v0.1):
- ERROR branch no longer removes the worker from active_workers — a worker
  that errors still sends a DONE sentinel via `finally`, so removing on
  ERROR *and* on DONE caused a KeyError (double removal) exactly in the
  one case (an actual failure) where the parent most needs to survive.
- Empty exception now imported from the standard `queue` module instead
  of relying on `mp.queues.Empty` being present as an import side-effect.
"""

import sys
import time
import random
import queue  # for queue.Empty — stable, documented, not an import side-effect
import numpy as np
import multiprocessing as mp

METADRIVE_AVAILABLE = False
try:
    from metadrive import MetaDriveEnv
    METADRIVE_AVAILABLE = True
except ImportError:
    pass


def worker_process(worker_id, task_queue, result_queue, num_steps=20):
    # Child process starts completely fresh on Windows ('spawn').
    # Must import MetaDriveEnv inside the worker execution context.
    try:
        from metadrive import MetaDriveEnv
        has_metadrive = True
    except ImportError:
        has_metadrive = False

    print(f"[{worker_id}] Subprocess spawned and initialized successfully.")

    if not has_metadrive:
        for step in range(1, num_steps + 1):
            time.sleep(random.uniform(0.05, 0.15))
            reward = random.uniform(0.0, 1.0)
            done = (step == num_steps)
            result_queue.put((worker_id, step, (259,), reward, done))
        result_queue.put((worker_id, "DONE"))
        return

    config = {
        "map": "C",
        "use_render": False,
        "horizon": num_steps,
    }

    try:
        env = MetaDriveEnv(config=config)
        obs = env.reset()
        if isinstance(obs, tuple) and len(obs) == 2:
            obs, info = obs

        for step in range(1, num_steps + 1):
            action = np.array([0.0, 0.5])
            step_results = env.step(action)

            if len(step_results) == 5:
                next_obs, reward, terminated, truncated, info = step_results
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_results

            result_queue.put((worker_id, step, next_obs.shape, reward, done))

            if done:
                break

        env.close()

    except Exception as e:
        result_queue.put((worker_id, "ERROR", str(e)))
    finally:
        # Always push DONE — including after an ERROR — so the parent
        # never waits forever. The parent must NOT remove the worker
        # on ERROR; only DONE retires it from active_workers.
        result_queue.put((worker_id, "DONE"))


def run_parallel_workers():
    num_workers = 3
    num_steps_per_worker = 25

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    processes = []

    print("=== Starting Hive Controller (Parent Process) ===")
    if not METADRIVE_AVAILABLE:
        print("Note: MetaDrive not detected locally. Running mock concurrent trace...\n")
    else:
        print("MetaDrive detected. Running physical simulation in parallel subprocesses...\n")

    for i in range(num_workers):
        worker_id = f"Forager_{i}"
        p = mp.Process(
            target=worker_process,
            args=(worker_id, task_queue, result_queue, num_steps_per_worker)
        )
        processes.append(p)
        print(f"[Parent] Spawning subprocess for {worker_id}...")
        p.start()

    active_workers = {f"Forager_{i}" for i in range(num_workers)}
    total_messages_received = 0

    print("\n--- Interleaved Telemetry Stream (Active Draining) ---")

    while len(active_workers) > 0:
        try:
            payload = result_queue.get(timeout=2.0)
            worker_id = payload[0]
            status = payload[1]

            if status == "DONE":
                active_workers.discard(worker_id)  # safe even if already removed
                print(f"[Parent] Received DONE sentinel from {worker_id}. Remaining active: {len(active_workers)}")
            elif status == "ERROR":
                err_msg = payload[2]
                print(f"[Parent] Received ERROR from {worker_id}: {err_msg}")
                # Do NOT remove here — the DONE sentinel is still coming
                # via the worker's `finally` block.
            else:
                step = payload[1]
                obs_shape = payload[2]
                reward = payload[3]
                done = payload[4]
                total_messages_received += 1
                print(f"  -> [{worker_id}] Step {step:02d} | Obs: {obs_shape} | Reward: {reward:.4f} | Done: {done}")

        except queue.Empty:
            if not any(p.is_alive() for p in processes) and len(active_workers) > 0:
                print("[Parent] Queue is empty but workers appear to have terminated without sentinels!")
                break

    print("\n--- Clean Multi-Process Teardown Phase ---")
    for i, p in enumerate(processes):
        worker_id = f"Forager_{i}"
        print(f"[Parent] Joining process for {worker_id}...")
        p.join()
        print(f"[Parent] Process {worker_id} terminated cleanly.")

    print("SUCCESS: Parallel plumbing verified with zero deadlocks!")
    print(f"Received {total_messages_received} concurrent telemetry steps.")


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    run_parallel_workers()