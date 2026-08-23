# Franka Panda MuJoCo Teleoperation → SmolVLA Dataset

Keyboard teleoperation of a Franka Emika Panda in MuJoCo, recording episodes in
`LeRobotDataset` format for fine-tuning SmolVLA, pushed to the Hugging Face Hub.

## Controls

| Key | Action |
|-----|--------|
| `w` / `s` | end-effector +X / -X |
| `a` / `d` | end-effector -Y / +Y |
| `e` / `c` | end-effector +Z / -Z |
| `q` | toggle gripper open/close |
| `r` | end episode, save if successful, reset |
| `x` | end episode, discard, reset |
| `esc` | quit |

**On `e`/`c`:** the original plan was w/a/s/d + q only. That gives purely planar
motion at fixed height — with it you literally cannot lift the cube off the
table, so no episode could ever succeed. Vertical control is added on `e`/`c`.
If you want the strict 5-key scheme, delete those two blocks in
`keyboard_controller.get_action()`, but expect the pick task to be impossible.

## Setup

1. `pip install -r requirements.txt`

2. Download the Panda model from
   [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda)
   into `assets/franka_emika_panda/`.

3. **Edit the downloaded `panda.xml`** — add a wrist camera and the IK
   reference site inside the `<body name="hand" ...>` block:

   ```xml
   <body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
     <camera name="wrist_cam" pos="0 0 0.05" xyaxes="0 -1 0  1 0 0"/>
     <site name="ee_site" pos="0 0 0.10" size="0.01" rgba="0 1 0 1"/>
     <inertial .../>
     ...
   ```

   This step is required — `panda.xml` is gitignored and re-downloaded fresh by
   anyone cloning this repo, so the edit does not travel with the project.

4. **If the model fails to load with a keyframe error**, delete the
   `<keyframe>...</keyframe>` block from `panda.xml`. Its `qpos` is sized for
   the bare arm; this scene adds a freejoint object (+7 qpos), so the lengths
   no longer match. `env.py` sets the home pose in code and does not need the
   keyframe.

5. Check the scene loads:
   ```
   python -m mujoco.viewer --mjcf=assets/franka_emika_panda/franka_panda_scene.xml
   ```

6. Record:  `python record_episode.py`

7. Convert and upload (after `hf auth login`, and after setting `HF_REPO_ID`):
   ```
   python convert_to_lerobot.py
   ```

## Data collected

- `observation.state` — 8 dims: 7 arm joint angles + gripper open/closed
- `action` — 4 dims: Δx, Δy, Δz + gripper open/closed
- `observation.images.front_cam`, `observation.images.wrist_cam` — 224×224 RGB

Episodes auto-save when the cube stays lifted ~1s, or on `r`. Failed attempts
are discarded rather than saved — a small clean dataset beats a large messy one.

## Dataset size

The SmolVLA authors report 50 episodes across 5 distinct cube positions
(10 each) worked for a single pick-place task, and that 25 was not enough.
`env.reset()` randomizes the cube position each episode, so just record ~50.

## Project structure

```
franka-mujoco-teleop/
├── record_episode.py              # main entry point
├── convert_to_lerobot.py          # raw .npz -> LeRobotDataset -> Hub
├── assets/
│   └── franka_emika_panda/        # from mujoco_menagerie (gitignored)
│       ├── panda.xml              #   ...edit per step 3
│       ├── assets/                #   meshes
│       └── franka_panda_scene.xml #   the scene — MUST sit here, next to
│                                  #   panda.xml (meshdir resolves relative to
│                                  #   the ROOT file across <include>)
└── teleop/
    ├── env.py                     # FrankaMujocoEnv
    ├── keyboard_controller.py     # KeyboardController
    ├── diff_ik.py                 # DiffIKSolver
    └── recorder.py                # EpisodeRecorder
```

## Architecture notes (for the self-study pass)

Four classes, one job each:

- **`FrankaMujocoEnv`** — encapsulates MuJoCo. Nothing outside it touches
  `model`/`data`. Note `__init__` resolving names→ids once: joint id, qpos
  index, and dof index are three different things, and confusing them is the
  classic MuJoCo beginner bug.
- **`KeyboardController`** — key state → action dict. No MuJoCo import at all.
  Note the edge-detection flags (`_q_was_down`): without them, holding `q` for
  half a second toggles the gripper ~8 times.
- **`DiffIKSolver`** — pure numpy, no MuJoCo. That's why `python -m
  teleop.diff_ik` can test it with a fake Jacobian and no simulator.
- **`EpisodeRecorder`** — buffers and dumps raw `.npz`. Knows nothing about
  LeRobot, so `convert_to_lerobot.py` can be re-run over existing recordings if
  you change the action representation.

`record_episode.py` builds one of each and wires them together — composition.
It contains almost no logic itself, which is why swapping `KeyboardController`
for a gamepad version would change nothing else.

Every `.copy()` in `env.py` and `recorder.py` is load-bearing: MuJoCo returns
views into buffers it overwrites next step.