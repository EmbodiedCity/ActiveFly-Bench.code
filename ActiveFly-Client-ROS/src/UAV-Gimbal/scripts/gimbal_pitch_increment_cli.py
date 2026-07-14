#!/usr/bin/env python3

import curses
import signal
import sys
import threading
import time
from typing import Any, Optional

import rospy
from gimbal.msg import GimbalCmd, GimbalState


class GimbalPitchIncrementCli:
    COLOR_TITLE = 1
    COLOR_HINT = 2
    COLOR_LABEL = 3
    COLOR_VALUE = 4
    COLOR_GOOD = 5
    COLOR_WARN = 6
    COLOR_ERROR = 7
    COLOR_GAUGE = 8
    COLOR_INPUT = 9

    GBC_STAT_MAP = {
        0: "undefined",
        1: "initializing",
        2: "stopped",
        3: "protect",
        4: "manual",
        5: "pointing",
    }

    CMD_STAT_MAP = {
        0: "running",
        1: "success",
        2: "failed",
    }

    INV_FLAG_MAP = {
        0: "hanging",
        1: "upright",
    }

    def __init__(self):
        rospy.init_node("gimbal_pitch_increment_cli")

        self.cmd_topic = str(rospy.get_param("~cmd_topic", "/gimbal/cmd"))
        self.state_topic = str(rospy.get_param("~state_topic", "/gimbal/state"))
        self.loop_hz = self._get_param_float("~loop_hz", 20.0)
        self.state_timeout_sec = self._get_param_float("~state_timeout_sec", 1.5)
        self.keepalive_sec = self._get_param_float("~keepalive_sec", 0.30)
        self.pitch_min_deg = self._get_param_float("~pitch_min_deg", self._get_param_float("~roll_min_deg", -90.0))
        self.pitch_max_deg = self._get_param_float("~pitch_max_deg", self._get_param_float("~roll_max_deg", 40.0))
        if self.pitch_min_deg > self.pitch_max_deg:
            rospy.logwarn(
                "pitch_min_deg(%.2f) > pitch_max_deg(%.2f), swap them",
                self.pitch_min_deg,
                self.pitch_max_deg,
            )
            self.pitch_min_deg, self.pitch_max_deg = self.pitch_max_deg, self.pitch_min_deg

        self.increment_step_deg = self._get_param_float("~increment_step_deg", 5.0)
        self.input_timeout_sec = self._get_param_float("~input_timeout_sec", 2.0)

        self.cmd_pub = rospy.Publisher(self.cmd_topic, GimbalCmd, queue_size=10)
        self.state_sub = rospy.Subscriber(self.state_topic, GimbalState, self._state_callback, queue_size=10)

        self.state_lock = threading.Lock()
        self.latest_state: Optional[GimbalState] = None
        self.latest_state_time = 0.0

        self.target_pitch_deg: Optional[float] = None
        self.last_cmd_pub_time = 0.0
        self.last_base_pitch_deg: Optional[float] = None
        self.last_delta_deg: Optional[float] = None
        self.last_unclamped_target_deg: Optional[float] = None
        self.last_action_note = "Waiting for numeric input"
        self.last_error_note = ""
        self.input_buffer = ""
        self.running = True
        self.colors_enabled = False

    @staticmethod
    def _get_param_float(name: str, default: float) -> float:
        raw: Any = rospy.get_param(name, default)
        if isinstance(raw, (int, float)):
            return float(raw)

        if isinstance(raw, str):
            try:
                return float(raw)
            except ValueError:
                rospy.logwarn("Parameter %s is not a valid float string (%s), fallback to %.3f", name, raw, default)
                return default

        rospy.logwarn("Parameter %s has unsupported type (%s), fallback to %.3f", name, type(raw).__name__, default)
        return default

    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, value))

    @staticmethod
    def _build_gauge(value_deg: float, min_deg: float, max_deg: float, width: int = 49) -> str:
        if width < 11:
            width = 11

        value_deg = max(min_deg, min(max_deg, value_deg))
        span = max_deg - min_deg
        if span <= 0:
            span = 1.0

        idx = int(round((value_deg - min_deg) / span * (width - 1)))
        idx = max(0, min(width - 1, idx))

        zero_idx = int(round((0.0 - min_deg) / span * (width - 1)))
        zero_idx = max(0, min(width - 1, zero_idx))

        chars = ["-"] * width
        chars[zero_idx] = "+"
        chars[idx] = "|"

        return "[{}]".format("".join(chars))

    def _state_callback(self, msg: GimbalState) -> None:
        with self.state_lock:
            self.latest_state = msg
            self.latest_state_time = time.monotonic()

    def _state_snapshot(self):
        with self.state_lock:
            if self.latest_state is None:
                return None, 0.0
            return self.latest_state, self.latest_state_time

    def _get_latest_state_pitch_deg(self) -> Optional[float]:
        state, state_time = self._state_snapshot()
        if state is None or state_time <= 0.0:
            return None

        if time.monotonic() - state_time > self.state_timeout_sec:
            return None

        return state.cam_angle_p * 0.01

    def _resolve_base_pitch_deg(self) -> Optional[float]:
        if self.target_pitch_deg is not None:
            return self.target_pitch_deg
        return self._get_latest_state_pitch_deg()

    def _publish_pitch_target(self, target_pitch_deg: float) -> None:
        cmd = GimbalCmd()
        cmd.ctrl_sw = 3
        cmd.cmd_value = 0
        cmd.op_value_p = int(round(target_pitch_deg * 100.0))
        cmd.op_value_r = 0
        cmd.op_value_y = 0
        self.cmd_pub.publish(cmd)
        self.target_pitch_deg = target_pitch_deg
        self.last_cmd_pub_time = time.monotonic()

    def _center_now(self) -> None:
        self.last_base_pitch_deg = self.target_pitch_deg if self.target_pitch_deg is not None else self._get_latest_state_pitch_deg()
        self.last_delta_deg = 0.0
        self.last_unclamped_target_deg = 0.0
        self._publish_pitch_target(0.0)
        self.last_action_note = "Center command sent: target pitch set to +0.00 deg"
        self.last_error_note = ""

    def _apply_delta(self, delta_deg: float) -> None:
        base_pitch_deg = self._resolve_base_pitch_deg()
        if base_pitch_deg is None:
            self.last_error_note = "No valid base angle yet. Wait for a fresh /gimbal/state sample or send after state is available."
            return

        unclamped_target_deg = base_pitch_deg + delta_deg
        target_pitch_deg = self._clamp(unclamped_target_deg, self.pitch_min_deg, self.pitch_max_deg)
        self._publish_pitch_target(target_pitch_deg)

        self.last_base_pitch_deg = base_pitch_deg
        self.last_delta_deg = delta_deg
        self.last_unclamped_target_deg = unclamped_target_deg
        if abs(target_pitch_deg - unclamped_target_deg) > 1e-6:
            self.last_action_note = (
                "Delta {:+.2f} deg applied from base {:+.2f} deg. Requested {:+.2f} deg, clamped to {:+.2f} deg".format(
                    delta_deg,
                    base_pitch_deg,
                    unclamped_target_deg,
                    target_pitch_deg,
                )
            )
        else:
            self.last_action_note = "Delta {:+.2f} deg applied from base {:+.2f} deg -> target {:+.2f} deg".format(
                delta_deg,
                base_pitch_deg,
                target_pitch_deg,
            )
        self.last_error_note = ""

    def _submit_buffer(self) -> None:
        raw = self.input_buffer.strip()
        self.input_buffer = ""

        if raw == "":
            self.last_error_note = "Input buffer is empty"
            return

        if raw in ("0", "+0", "-0", "0.0", "+0.0", "-0.0"):
            self._center_now()
            return

        try:
            delta_deg = float(raw)
        except ValueError:
            self.last_error_note = "Invalid input '{}'. Enter a signed number, 0 to center, or q to quit.".format(raw)
            return

        self._apply_delta(delta_deg)

    def _process_key(self, key: int) -> None:
        if key == -1:
            return

        if key in (ord("q"), ord("Q")):
            self.running = False
            return

        if key in (10, 13, curses.KEY_ENTER):
            self._submit_buffer()
            return

        if key in (ord("c"), ord("C")):
            self.input_buffer = ""
            self._center_now()
            return

        if key in (curses.KEY_BACKSPACE, 127, 8):
            self.input_buffer = self.input_buffer[:-1]
            return

        if key in (ord("["),):
            self._apply_delta(-self.increment_step_deg)
            return

        if key in (ord("]"),):
            self._apply_delta(self.increment_step_deg)
            return

        if 32 <= key <= 126:
            ch = chr(key)
            if ch in "0123456789+-.":
                self.input_buffer += ch
            else:
                self.last_error_note = "Unsupported key '{}'. Type a number, Enter to send, C to center, Q to quit.".format(ch)

    def _init_colors(self) -> None:
        self.colors_enabled = False
        if not curses.has_colors():
            return

        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(self.COLOR_TITLE, curses.COLOR_CYAN, -1)
        curses.init_pair(self.COLOR_HINT, curses.COLOR_BLUE, -1)
        curses.init_pair(self.COLOR_LABEL, curses.COLOR_WHITE, -1)
        curses.init_pair(self.COLOR_VALUE, curses.COLOR_CYAN, -1)
        curses.init_pair(self.COLOR_GOOD, curses.COLOR_GREEN, -1)
        curses.init_pair(self.COLOR_WARN, curses.COLOR_YELLOW, -1)
        curses.init_pair(self.COLOR_ERROR, curses.COLOR_RED, -1)
        curses.init_pair(self.COLOR_GAUGE, curses.COLOR_MAGENTA, -1)
        curses.init_pair(self.COLOR_INPUT, curses.COLOR_BLACK, curses.COLOR_CYAN)
        self.colors_enabled = True

    def _color_attr(self, color_id: int, fallback: int = 0) -> int:
        if self.colors_enabled:
            return curses.color_pair(color_id)
        return fallback

    def _draw_text(self, stdscr, row: int, col: int, text: str, attr: int = 0) -> None:
        max_y, max_x = stdscr.getmaxyx()
        if row < 0 or row >= max_y or col >= max_x:
            return
        if col < 0:
            col = 0
        stdscr.addnstr(row, col, text, max_x - col - 1, attr)

    def _render(self, stdscr) -> None:
        stdscr.erase()

        state, state_time = self._state_snapshot()
        now = time.monotonic()
        state_age = now - state_time if state_time > 0.0 else float("inf")
        max_y, max_x = stdscr.getmaxyx()

        row = 0
        self._draw_text(stdscr, row, 0, "Gimbal Pitch Increment CLI", self._color_attr(self.COLOR_TITLE, curses.A_BOLD) | curses.A_BOLD)
        row += 1
        self._draw_text(
            stdscr,
            row,
            0,
            "Type a signed delta and press Enter | 0 or C: center | Q: quit | [: -step | ]: +step",
            self._color_attr(self.COLOR_HINT),
        )
        row += 1
        self._draw_text(
            stdscr,
            row,
            0,
            "Realtime fullscreen UI with colored status sections and no scrolling terminal output.",
            self._color_attr(self.COLOR_HINT),
        )
        row += 2

        self._draw_text(stdscr, row, 0, "Step size:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
        self._draw_text(stdscr, row, 11, "{:5.2f} deg".format(self.increment_step_deg), self._color_attr(self.COLOR_VALUE))
        self._draw_text(stdscr, row, 26, "Limit:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
        self._draw_text(
            stdscr,
            row,
            33,
            "[{:+5.1f}, {:+5.1f}] deg".format(self.pitch_min_deg, self.pitch_max_deg),
            self._color_attr(self.COLOR_VALUE),
        )
        row += 1
        self._draw_text(stdscr, row, 0, "cmd topic:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
        self._draw_text(stdscr, row, 11, self.cmd_topic, self._color_attr(self.COLOR_VALUE))
        self._draw_text(stdscr, row, min(11 + len(self.cmd_topic) + 3, max_x - 1), "state topic:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
        self._draw_text(stdscr, row, min(24 + len(self.cmd_topic), max_x - 1), self.state_topic, self._color_attr(self.COLOR_VALUE))
        row += 2

        target_text = "None" if self.target_pitch_deg is None else "{:+6.2f} deg".format(self.target_pitch_deg)
        base_text = "None" if self.last_base_pitch_deg is None else "{:+6.2f} deg".format(self.last_base_pitch_deg)
        delta_text = "None" if self.last_delta_deg is None else "{:+6.2f} deg".format(self.last_delta_deg)
        req_text = "None" if self.last_unclamped_target_deg is None else "{:+6.2f} deg".format(self.last_unclamped_target_deg)
        self._draw_text(stdscr, row, 0, "Target pitch:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
        self._draw_text(stdscr, row, 14, target_text, self._color_attr(self.COLOR_GOOD if self.target_pitch_deg is not None else self.COLOR_WARN))
        row += 1
        self._draw_text(stdscr, row, 0, "Last base:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
        self._draw_text(stdscr, row, 11, base_text, self._color_attr(self.COLOR_VALUE))
        self._draw_text(stdscr, row, 27, "Last delta:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
        self._draw_text(stdscr, row, 39, delta_text, self._color_attr(self.COLOR_VALUE))
        self._draw_text(stdscr, row, 55, "Requested:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
        self._draw_text(stdscr, row, 66, req_text, self._color_attr(self.COLOR_VALUE))
        row += 2

        if state is None:
            self._draw_text(stdscr, row, 0, "State:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
            self._draw_text(stdscr, row, 7, "waiting for {} ...".format(self.state_topic), self._color_attr(self.COLOR_WARN))
            row += 1
            self._draw_text(
                stdscr,
                row,
                0,
                self._build_gauge(self.target_pitch_deg if self.target_pitch_deg is not None else 0.0, self.pitch_min_deg, self.pitch_max_deg),
                self._color_attr(self.COLOR_GAUGE),
            )
            row += 1
        else:
            cam_p = state.cam_angle_p * 0.01
            cam_r = state.cam_angle_r * 0.01
            cam_y = state.cam_angle_y * 0.01
            mtr_p = state.mtr_angle_p * 0.01
            mtr_r = state.mtr_angle_r * 0.01
            mtr_y = state.mtr_angle_y * 0.01
            rate_p = state.cam_rate_p * 0.01
            rate_r = state.cam_rate_r * 0.01
            rate_y = state.cam_rate_y * 0.01

            gbc_stat_text = self.GBC_STAT_MAP.get(state.gbc_stat, "unknown")
            cmd_stat_text = self.CMD_STAT_MAP.get(state.cmd_stat, "unknown")
            inv_text = self.INV_FLAG_MAP.get(state.inv_fag, "unknown")
            tca_text = "ready" if state.tca_fag else "not_ready"
            freshness = "fresh" if state_age < 1.0 else "stale"
            freshness_color = self.COLOR_GOOD if freshness == "fresh" else self.COLOR_WARN

            self._draw_text(stdscr, row, 0, "State age:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
            self._draw_text(stdscr, row, 11, "{:5.2f}s".format(state_age), self._color_attr(freshness_color))
            self._draw_text(stdscr, row, 19, "({})".format(freshness), self._color_attr(freshness_color))
            row += 1
            self._draw_text(stdscr, row, 0, "Gimbal mode:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
            self._draw_text(stdscr, row, 13, "{}({})".format(gbc_stat_text, state.gbc_stat), self._color_attr(self.COLOR_VALUE))
            self._draw_text(stdscr, row, 33, "Cmd:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
            self._draw_text(stdscr, row, 38, "{}({}) value:{}".format(cmd_stat_text, state.cmd_stat, state.cmd_value), self._color_attr(self.COLOR_VALUE))
            row += 1
            self._draw_text(stdscr, row, 0, "Mount:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
            self._draw_text(stdscr, row, 7, inv_text, self._color_attr(self.COLOR_VALUE))
            self._draw_text(stdscr, row, 20, "Temp:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
            self._draw_text(stdscr, row, 26, tca_text, self._color_attr(self.COLOR_GOOD if state.tca_fag else self.COLOR_WARN))
            row += 1
            self._draw_text(stdscr, row, 0, "Cam angle (deg) p/r/y:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
            self._draw_text(stdscr, row, 23, "{:+7.2f}  {:+7.2f}  {:+7.2f}".format(cam_p, cam_r, cam_y), self._color_attr(self.COLOR_VALUE))
            row += 1
            self._draw_text(stdscr, row, 0, "Motor angle(deg) p/r/y:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
            self._draw_text(stdscr, row, 24, "{:+7.2f}  {:+7.2f}  {:+7.2f}".format(mtr_p, mtr_r, mtr_y), self._color_attr(self.COLOR_VALUE))
            row += 1
            self._draw_text(stdscr, row, 0, "Cam rate  (deg/s) p/r/y:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
            self._draw_text(stdscr, row, 25, "{:+7.2f}  {:+7.2f}  {:+7.2f}".format(rate_p, rate_r, rate_y), self._color_attr(self.COLOR_VALUE))
            row += 2
            self._draw_text(stdscr, row, 0, "Pitch angle gauge (camera pitch)", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
            row += 1
            self._draw_text(stdscr, row, 0, self._build_gauge(cam_p, self.pitch_min_deg, self.pitch_max_deg), self._color_attr(self.COLOR_GAUGE))
            row += 1

        row += 1
        self._draw_text(stdscr, row, 0, "Last action:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
        self._draw_text(stdscr, row, 13, self.last_action_note, self._color_attr(self.COLOR_GOOD))
        row += 1
        self._draw_text(stdscr, row, 0, "Last error:", self._color_attr(self.COLOR_LABEL, curses.A_BOLD))
        error_text = self.last_error_note if self.last_error_note else "<none>"
        error_color = self.COLOR_ERROR if self.last_error_note else self.COLOR_HINT
        self._draw_text(stdscr, row, 12, error_text, self._color_attr(error_color))

        input_row = max_y - 2
        prompt = "Input Buffer: {}".format(self.input_buffer if self.input_buffer else "<empty>")
        self._draw_text(stdscr, input_row, 0, " " * max(0, max_x - 1), self._color_attr(self.COLOR_INPUT))
        self._draw_text(stdscr, input_row, 0, prompt, self._color_attr(self.COLOR_INPUT) | curses.A_BOLD)
        self._draw_text(
            stdscr,
            max_y - 1,
            0,
            "Enter=submit | Backspace=edit | C=center | Q=quit | [ ]=step",
            self._color_attr(self.COLOR_HINT),
        )

        stdscr.refresh()

    def _loop(self, stdscr) -> None:
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.keypad(True)
        self._init_colors()

        rate = rospy.Rate(self.loop_hz)
        while self.running and not rospy.is_shutdown():
            key = stdscr.getch()
            self._process_key(key)

            now = time.monotonic()
            if self.target_pitch_deg is not None and now - self.last_cmd_pub_time > self.keepalive_sec:
                self._publish_pitch_target(self.target_pitch_deg)

            self._render(stdscr)
            rate.sleep()

    def run(self) -> None:
        def _sigint_handler(signum, frame):
            del signum, frame
            self.running = False

        signal.signal(signal.SIGINT, _sigint_handler)
        curses.wrapper(self._loop)


def main() -> None:
    try:
        GimbalPitchIncrementCli().run()
    except rospy.ROSInterruptException:
        pass
    except curses.error:
        sys.stderr.write("Failed to initialize curses UI. Run this script in an interactive terminal.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
