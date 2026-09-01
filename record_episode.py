"""
FILE 5: record_episode.py  —  MAIN ENTRY POINT

    python record_episode.py

Controls
    w / s : end-effector  +X / -X
    a / d : end-effector  -Y / +Y
    e / c : end-effector  +Z / -Z
    q     : toggle gripper open/close
    r     : end episode, SAVE if successful, reset
    x     : end episode, DISCARD, reset
    esc   : quit

NOTE ON THE PERSISTENT TARGET
    The EE target is held in ee_target_pos and deltas ACCUMULATE onto it.
    It is deliberately NOT re-derived from the measured EE pose each tick.

    The Panda's actuators are pure PD (force = kp*(ctrl - qpos) - kd*qvel)
    with no gravity-compensation term, so there is always a small steady-state
    error under load. If the target were re-anchored to the measured (already
    sagging) pose every tick, that error would compound and the arm would
    droop without bound. Against a fixed target the sag stays ~0.01 rad and
    the IK actively corrects it.

    The leash clamp below stops the target running away when the arm is
    physically blocked (e.g. pressing down into the table).
"""

import time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

from teleop.env import FrankaMujocoEnv
from teleop.keyboard_controller import KeyboardController
from teleop.diff_ik import DiffIKSolver
from teleop.recorder import EpisodeRecorder


# ==========================================================================
# Config
# ==========================================================================
XML_PATH = "assets/franka_emika_panda/franka_panda_scene.xml"
CAMERA_NAMES = ["front_cam", "wrist_cam"]
RENDER_SIZE = (224, 224)

CONTROL_HZ = 15          # MUST match FPS in convert_to_lerobot.py
SAVE_DIR = "recorded_episodes_fresh"
TASK_INSTRUCTION = "pick up the red cube"

STEP_SIZE = 0.01         # metres per key per tick
IK_DAMPING = 0.05
MAX_LEASH = 0.05         # max allowed gap between target and actual EE (m)

TABLE_TOP_Z = 0.0
LIFT_THRESHOLD = 0.05

AUTO_SAVE_ON_SUCCESS = True
SUCCESS_HOLD_TICKS = 15
TARGET_EPISODES = 50


def main():
    env = FrankaMujocoEnv(
        xml_path=XML_PATH,
        camera_names=CAMERA_NAMES,
        render_size=RENDER_SIZE,
        control_hz=CONTROL_HZ,
        table_top_z=TABLE_TOP_Z,
        lift_threshold=LIFT_THRESHOLD,
    )
    controller = KeyboardController(step_size=STEP_SIZE)
    ik_solver = DiffIKSolver(damping=IK_DAMPING)
    recorder = EpisodeRecorder(save_dir=SAVE_DIR, task_instruction=TASK_INSTRUCTION)

    joint_limits = env.get_joint_limits()
    g_lo, g_hi = env.gripper_ctrlrange

    # Resume safely when recording is restarted; never overwrite an existing
    # demonstration. Gaps in numbering are allowed.
    existing_episode_paths = sorted(Path(SAVE_DIR).glob("episode_*.npz"))
    existing_indices = []
    for path in existing_episode_paths:
        try:
            existing_indices.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            pass
    episode_index = max(existing_indices, default=-1) + 1
    total_saved = len(existing_episode_paths)
    success_ticks = 0
    dt = 1.0 / CONTROL_HZ

    env.reset()
    # persistent target, seeded from the home pose
    ee_target_pos, ee_target_quat = env.get_ee_pose()

    print("=" * 62)
    print(" w/s: +X/-X   a/d: -Y/+Y   e/c: +Z/-Z")
    print(" q: gripper   r: save+reset   x: discard+reset   esc: quit")
    print(f" start EE pos: {np.round(ee_target_pos, 3)}")
    print(f" saved episodes: {total_saved}/{TARGET_EPISODES}; next index: {episode_index}")
    print("=" * 62)

    def do_reset():
        nonlocal success_ticks, ee_target_pos, ee_target_quat
        env.reset()
        # re-seed the target, or the arm will lunge toward the old one
        ee_target_pos, ee_target_quat = env.get_ee_pose()
        success_ticks = 0

    def end_episode(save: bool):
        nonlocal episode_index, total_saved
        n = recorder.num_frames()
        if save and n > 0:
            recorder.save_episode(episode_index, success=True)
            print(f"[episode {episode_index}] SAVED  ({n} frames)")
            episode_index += 1
            total_saved += 1
            print(f"[progress] {total_saved}/{TARGET_EPISODES} episodes")
        else:
            recorder.discard_episode()
            print(f"[discarded] ({n} frames)")
        do_reset()

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running() and total_saved < TARGET_EPISODES:
                loop_start = time.time()

                if controller.should_quit():
                    break

                action = controller.get_action()
                current_pos, current_quat = env.get_ee_pose()
                current_joints = env.get_joint_positions()

                # --- observation BEFORE stepping (standard IL convention) ---
                images = env.render_all_cameras()
                state = np.concatenate([
                    current_joints,
                    [1.0 if action["gripper_closed"] else 0.0],
                ]).astype(np.float32)
                action_vec = np.concatenate([
                    action["delta_pos"],
                    [1.0 if action["gripper_closed"] else 0.0],
                ]).astype(np.float32)
                recorder.add_frame(images, state, action_vec)

                # --- accumulate onto the PERSISTENT target ------------------
                ee_target_pos = ee_target_pos + action["delta_pos"]

                # leash: keep the target within reach of the actual EE, so it
                # can't run away while the arm is blocked by the table
                gap = ee_target_pos - current_pos
                gap_norm = np.linalg.norm(gap)
                if gap_norm > MAX_LEASH:
                    ee_target_pos = current_pos + gap * (MAX_LEASH / gap_norm)

                # --- IK ------------------------------------------------------
                q_target = ik_solver.solve(
                    current_pos, current_quat,
                    ee_target_pos, ee_target_quat,
                    env.get_jacobian(), current_joints,
                    joint_limits,
                )
                gripper_ctrl = g_lo if action["gripper_closed"] else g_hi
                env.step(q_target, gripper_ctrl)

                viewer.sync()

                # --- episode boundaries --------------------------------------
                if controller.should_reset():
                    end_episode(save=env.is_grasped())
                elif controller.should_discard():
                    end_episode(save=False)
                elif AUTO_SAVE_ON_SUCCESS:
                    success_ticks = success_ticks + 1 if env.is_grasped() else 0
                    if success_ticks >= SUCCESS_HOLD_TICKS:
                        end_episode(save=True)

                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

    finally:
        controller.close()
        env.close()
        print(f"\nDone. {total_saved}/{TARGET_EPISODES} episodes saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
