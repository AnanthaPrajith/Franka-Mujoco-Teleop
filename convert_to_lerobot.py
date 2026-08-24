"""
FILE 6: convert_to_lerobot.py

    hf auth login          # once, first
    python convert_to_lerobot.py

Reads recorded_episodes/*.npz, builds a LeRobotDataset, pushes to the Hub.

NOTE: LeRobot's API has shifted between releases. If the import or the
create()/add_frame() signatures fail, check the docs for YOUR installed
version (`pip show lerobot`) — the data handling below stays the same, only
the call signatures change.
"""

from pathlib import Path
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ==========================================================================
# Config — must match record_episode.py
# ==========================================================================
HF_REPO_ID = "Prajith7roboq/franka_pick_mujoco"   # <-- CHANGE THIS
RAW_EPISODES_DIR = "recorded_episodes"
FPS = 15
CAMERA_NAMES = ["front_cam", "wrist_cam"]
IMAGE_SHAPE = (224, 224, 3)

STATE_DIM = 8     # 7 arm joints + 1 gripper
ACTION_DIM = 4    # dx, dy, dz + gripper

PUSH_TO_HUB = True
SKIP_FAILED = True


def build_dataset() -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": [f"joint{i}" for i in range(1, 8)] + ["gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": ["dx", "dy", "dz", "gripper"],
        },
    }
    for cam in CAMERA_NAMES:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        }

    return LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=features,
        robot_type="franka_panda",
    )


def load_raw_episode(npz_path: Path) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    return {
        "states": z["states"],
        "actions": z["actions"],
        "success": bool(z["success"]),
        "task_instruction": str(z["task_instruction"]),
        "images": {cam: z[f"images_{cam}"] for cam in CAMERA_NAMES if f"images_{cam}" in z},
    }


def add_episode_to_dataset(dataset: LeRobotDataset, ep: dict):
    n = len(ep["states"])
    for i in range(n):
        frame = {
            "observation.state": ep["states"][i].astype(np.float32),
            "action": ep["actions"][i].astype(np.float32),
            # LeRobot 0.4.x expects the natural-language task in each frame.
            "task": ep["task_instruction"],
        }
        for cam, arr in ep["images"].items():
            frame[f"observation.images.{cam}"] = arr[i]
        dataset.add_frame(frame)

    # This is what actually writes the episode (encodes video, writes parquet).
    # Forget it and you get an empty dataset with no error.
    dataset.save_episode()


def main():
    raw_dir = Path(RAW_EPISODES_DIR)
    episode_files = sorted(raw_dir.glob("episode_*.npz"))
    print(f"Found {len(episode_files)} raw episodes in {raw_dir}/")

    if not episode_files:
        print("Nothing to convert. Record some episodes first.")
        return

    if len(episode_files) < 50:
        print(f"NOTE: only {len(episode_files)} episodes. The SmolVLA authors "
              f"report 50 (across 5 object positions) worked and 25 did not.")

    dataset = build_dataset()

    n_added = 0
    for path in episode_files:
        ep = load_raw_episode(path)
        if SKIP_FAILED and not ep["success"]:
            print(f"  skip  {path.name} (failed)")
            continue
        add_episode_to_dataset(dataset, ep)
        n_added += 1
        print(f"  added {path.name} ({len(ep['states'])} frames)")

    print(f"\nAdded {n_added} episodes.")

    if PUSH_TO_HUB:
        print(f"Pushing to {HF_REPO_ID} ...")
        dataset.push_to_hub()
        print("Done. Check the dataset viewer on huggingface.co before training.")
    else:
        print("PUSH_TO_HUB is False — dataset built locally only.")


if __name__ == "__main__":
    main()
