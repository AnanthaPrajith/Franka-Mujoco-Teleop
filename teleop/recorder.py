"""
FILE 4: teleop/recorder.py  —  EpisodeRecorder

Buffers frames during an episode, writes raw .npz to disk when it ends.
Deliberately knows nothing about LeRobot — that conversion happens separately
in convert_to_lerobot.py, so you can re-run it without re-teleoperating.
"""

import numpy as np
from pathlib import Path


class EpisodeRecorder:

    def __init__(self, save_dir: str, task_instruction: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.task_instruction = task_instruction
        self._frames = []

    # ----------------------------------------------------------------------
    def add_frame(self, images: dict, state: np.ndarray, action: np.ndarray):
        # .copy() is essential: MuJoCo returns views into buffers it overwrites
        # on the next step. Without copying, every frame ends up identical.
        self._frames.append({
            "images": {k: np.asarray(v).copy() for k, v in images.items()},
            "state": np.asarray(state, dtype=np.float32).copy(),
            "action": np.asarray(action, dtype=np.float32).copy(),
        })

    def num_frames(self) -> int:
        return len(self._frames)

    # ----------------------------------------------------------------------
    def save_episode(self, episode_index: int, success: bool) -> bool:
        if not self._frames:
            return False

        states = np.stack([f["state"] for f in self._frames])
        actions = np.stack([f["action"] for f in self._frames])

        image_arrays = {}
        for cam in self._frames[0]["images"].keys():
            image_arrays[f"images_{cam}"] = np.stack(
                [f["images"][cam] for f in self._frames]
            )

        path = self.save_dir / f"episode_{episode_index:04d}.npz"
        np.savez_compressed(
            path,
            states=states,
            actions=actions,
            success=np.array(success),
            task_instruction=np.array(self.task_instruction),
            **image_arrays,
        )
        self._frames = []
        return True

    # ----------------------------------------------------------------------
    def discard_episode(self):
        self._frames = []