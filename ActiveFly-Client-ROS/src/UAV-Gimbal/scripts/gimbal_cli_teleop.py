#!/usr/bin/env python3

import curses
import locale
import signal
import sys
import threading
import time
from typing import Any

import rospy
from gimbal.msg import GimbalCmd, GimbalState


class GimbalCliTeleop:
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
        locale.setlocale(locale.LC_ALL, "")
        rospy.init_node("gimbal_cli_teleop")

        self.cmd_topic = rospy.get_param("~cmd_topic", "/gimbal/cmd")
        self.state_topic = rospy.get_param("~state_topic", "/gimbal/state")

        self.loop_hz = self._get_param_float("~loop_hz", 30.0)
        self.pitch_min_deg = self._get_param_float("~pitch_min_deg", self._get_param_float("~roll_min_deg", -90.0))
        self.pitch_max_deg = self._get_param_float("~pitch_max_deg", self._get_param_float("~roll_max_deg", 40.0))
        if self.pitch_min_deg > self.pitch_max_deg:
            rospy.logwarn("pitch_min_deg(%.2f) > pitch_max_deg(%.2f), swap them", self.pitch_min_deg, self.pitch_max_deg)
            self.pitch_min_deg, self.pitch_max_deg = self.pitch_max_deg, self.pitch_min_deg

        self.pitch_rate_deg_per_sec = self._get_param_float(
            "~pitch_rate_deg_per_sec", self._get_param_float("~roll_rate_deg_per_sec", 30.0)
        )
        self.key_latch_sec = self._get_param_float("~key_latch_sec", 0.18)
        self.keepalive_sec = self._get_param_float("~keepalive_sec", 0.30)
        self.center_on_exit = bool(rospy.get_param("~center_on_exit", True))

        self.target_pitch_deg = self._get_param_float(
            "~init_target_pitch_deg", self._get_param_float("~init_target_roll_deg", 0.0)
        )
        self.target_pitch_deg = self._clamp(self.target_pitch_deg, self.pitch_min_deg, self.pitch_max_deg)

        self.cmd_pub = rospy.Publisher(self.cmd_topic, GimbalCmd, queue_size=10)
        self.state_sub = rospy.Subscriber(self.state_topic, GimbalState, self._state_callback, queue_size=10)

        self.state_lock = threading.Lock()
        self.latest_state = None
        self.latest_state_time = 0.0

        self.running = True
        self.pitch_direction = 0
        self.last_key_time = 0.0
        self.last_cmd_pub_time = 0.0

    @staticmethod
    def _clamp(value, min_val, max_val):
        return max(min_val, min(max_val, value))

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

    def _state_callback(self, msg):
        with self.state_lock:
            self.latest_state = msg
            self.latest_state_time = time.monotonic()

    def _publish_pitch_cmd(self):
        cmd = GimbalCmd()
        cmd.ctrl_sw = 3
        cmd.cmd_value = 0
        cmd.op_value_p = int(round(self.target_pitch_deg * 100.0))
        cmd.op_value_r = 0
        cmd.op_value_y = 0
        self.cmd_pub.publish(cmd)
        self.last_cmd_pub_time = time.monotonic()

    def _publish_mode_cmd(self, cmd_value):
        cmd = GimbalCmd()
        cmd.ctrl_sw = 1
        cmd.cmd_value = int(cmd_value)
        cmd.op_value_p = 0
        cmd.op_value_r = 0
        cmd.op_value_y = 0
        self.cmd_pub.publish(cmd)

    def _center_now(self):
        self.target_pitch_deg = 0.0
        self.pitch_direction = 0
        self._publish_pitch_cmd()

    @staticmethod
    def _build_gauge(value_deg, min_deg, max_deg, width=49):
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

    def _state_snapshot(self):
        with self.state_lock:
            if self.latest_state is None:
                return None, 0.0
            return self.latest_state, self.latest_state_time

    def _render(self, stdscr):
        stdscr.erase()

        state, state_time = self._state_snapshot()
        now = time.monotonic()
        state_age = now - state_time if state_time > 0.0 else float("inf")

        lines = []
        lines.append("Gimbal CLI Teleop (Pitch Axis)")
        lines.append("W/Up: pitch+  S/Down: pitch-  C: center  M:manual(cmd=4)  T:start(cmd=2)  X:stop(cmd=3)")
        lines.append("D: speed+  A: speed-  Q: quit")
        lines.append("")
        lines.append("Target pitch: {:+6.2f} deg | Rate: {:5.1f} deg/s | Limit: [{:+4.1f}, {:+4.1f}] deg".format(
            self.target_pitch_deg,
            self.pitch_rate_deg_per_sec,
            self.pitch_min_deg,
            self.pitch_max_deg,
        ))

        if state is None:
            lines.append("State: waiting for {} ...".format(self.state_topic))
            lines.append(self._build_gauge(0.0, self.pitch_min_deg, self.pitch_max_deg))
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

            lines.append("State age: {:5.2f}s ({})".format(state_age, freshness))
            lines.append("Gimbal mode: {}({}) | Cmd: {}({}) value:{} | Mount:{} | Temp:{}".format(
                gbc_stat_text,
                state.gbc_stat,
                cmd_stat_text,
                state.cmd_stat,
                state.cmd_value,
                inv_text,
                tca_text,
            ))
            lines.append("Cam angle (deg) p/r/y: {:+7.2f}  {:+7.2f}  {:+7.2f}".format(cam_p, cam_r, cam_y))
            lines.append("Motor angle(deg) p/r/y: {:+7.2f}  {:+7.2f}  {:+7.2f}".format(mtr_p, mtr_r, mtr_y))
            lines.append("Cam rate  (deg/s) p/r/y: {:+7.2f}  {:+7.2f}  {:+7.2f}".format(rate_p, rate_r, rate_y))
            lines.append("")
            lines.append("Pitch angle gauge (camera pitch)")
            lines.append(self._build_gauge(cam_p, self.pitch_min_deg, self.pitch_max_deg))

        max_y, max_x = stdscr.getmaxyx()
        for row, text in enumerate(lines):
            if row >= max_y:
                break
            stdscr.addnstr(row, 0, text, max_x - 1)

        stdscr.refresh()

    def _process_keys(self, stdscr, now):
        while True:
            key = stdscr.getch()
            if key == -1:
                break

            if key in (ord("q"), ord("Q")):
                self.running = False
                return

            if key in (ord("s"), ord("S"), curses.KEY_DOWN):
                self.pitch_direction = -1
                self.last_key_time = now
                continue

            if key in (ord("w"), ord("W"), curses.KEY_UP):
                self.pitch_direction = 1
                self.last_key_time = now
                continue

            if key in (ord("c"), ord("C")):
                self._center_now()
                continue

            if key in (ord("m"), ord("M")):
                self._publish_mode_cmd(4)
                continue

            if key in (ord("t"), ord("T")):
                self._publish_mode_cmd(2)
                continue

            if key in (ord("x"), ord("X")):
                self._publish_mode_cmd(3)
                continue

            if key in (ord("d"), ord("D")):
                self.pitch_rate_deg_per_sec = min(120.0, self.pitch_rate_deg_per_sec + 5.0)
                continue

            if key in (ord("a"), ord("A")):
                self.pitch_rate_deg_per_sec = max(1.0, self.pitch_rate_deg_per_sec - 5.0)
                continue

    def _loop(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.keypad(True)

        self._publish_mode_cmd(4)
        self._publish_pitch_cmd()

        last_time = time.monotonic()
        rate = rospy.Rate(self.loop_hz)

        while self.running and not rospy.is_shutdown():
            now = time.monotonic()
            dt = now - last_time
            last_time = now

            self._process_keys(stdscr, now)

            if now - self.last_key_time > self.key_latch_sec:
                self.pitch_direction = 0

            if self.pitch_direction != 0:
                self.target_pitch_deg += self.pitch_direction * self.pitch_rate_deg_per_sec * dt
                self.target_pitch_deg = self._clamp(self.target_pitch_deg, self.pitch_min_deg, self.pitch_max_deg)
                self._publish_pitch_cmd()
            elif now - self.last_cmd_pub_time > self.keepalive_sec:
                self._publish_pitch_cmd()

            self._render(stdscr)
            rate.sleep()

    def run(self):
        def _sigint_handler(signum, frame):
            del signum, frame
            self.running = False

        signal.signal(signal.SIGINT, _sigint_handler)

        try:
            curses.wrapper(self._loop)
        finally:
            if self.center_on_exit and not rospy.is_shutdown():
                self._center_now()
                rospy.sleep(0.05)


def main():
    try:
        app = GimbalCliTeleop()
        app.run()
    except rospy.ROSInterruptException:
        pass
    except curses.error:
        sys.stderr.write("Failed to initialize curses UI. Run this script in an interactive terminal.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
