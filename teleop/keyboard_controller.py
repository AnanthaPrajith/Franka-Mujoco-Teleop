"""
FILE 2: teleop/keyboard_controller.py  —  KeyboardController

Turns keyboard state into a Cartesian delta + gripper command.
Knows nothing about MuJoCo, IK, or datasets.
"""

import numpy as np
from pynput import keyboard


class KeyboardController:

    def __init__(self, step_size: float = 0.01):
        self.step_size = step_size

        self._pressed = set()
        self.gripper_closed = False

        # edge-detection state: distinguishes "held" from "just pressed"
        self._q_was_down = False
        self._r_was_down = False
        self._x_was_down = False

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

    # ----------------------------------------------------------------------
    def _on_press(self, key):
        try:
            self._pressed.add(key.char.lower())
        except AttributeError:
            # special keys have no .char
            if key == keyboard.Key.esc:
                self._pressed.add("esc")

    def _on_release(self, key):
        try:
            self._pressed.discard(key.char.lower())
        except AttributeError:
            if key == keyboard.Key.esc:
                self._pressed.discard("esc")

    # ----------------------------------------------------------------------
    def get_action(self) -> dict:
        """{"delta_pos": (3,) array, "gripper_closed": bool}"""
        delta = np.zeros(3)
        p = self._pressed

        if "w" in p:
            delta[0] += self.step_size
        if "s" in p:
            delta[0] -= self.step_size
        if "a" in p:
            delta[1] -= self.step_size
        if "d" in p:
            delta[1] += self.step_size

        # bonus vertical control (not in the original 5-key spec, but a pick
        # task is far easier with it; ignore these keys if you don't want it)
        if "e" in p:
            delta[2] += self.step_size
        if "c" in p:
            delta[2] -= self.step_size

        # edge-triggered gripper toggle
        q_down = "q" in p
        if q_down and not self._q_was_down:
            self.gripper_closed = not self.gripper_closed
        self._q_was_down = q_down

        return {"delta_pos": delta, "gripper_closed": self.gripper_closed}

    # ----------------------------------------------------------------------
    def should_quit(self) -> bool:
        return "esc" in self._pressed

    def should_reset(self) -> bool:
        """True exactly once per fresh 'r' press (save + reset)."""
        down = "r" in self._pressed
        fired = down and not self._r_was_down
        self._r_was_down = down
        return fired

    def should_discard(self) -> bool:
        """True exactly once per fresh 'x' press (discard + reset)."""
        down = "x" in self._pressed
        fired = down and not self._x_was_down
        self._x_was_down = down
        return fired

    # ----------------------------------------------------------------------
    def close(self):
        try:
            self._listener.stop()
        except Exception:
            pass


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    ctrl = KeyboardController(step_size=0.01)
    print("Press w/a/s/d (+ e/c for up/down), q toggles gripper, esc quits.")
    try:
        while not ctrl.should_quit():
            a = ctrl.get_action()
            print(f"delta={np.round(a['delta_pos'], 4)}  gripper_closed={a['gripper_closed']}")
            time.sleep(0.15)
    finally:
        ctrl.close()