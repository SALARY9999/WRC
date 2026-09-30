"""The log field contract: append-only, and only where it was decided.

``tools/check_chain.sh`` greps ``backend.log`` for substrings
(``inference sent: control_flag=``, ``pipeline cycle failed``, ...) and
operators read the rest by eye.  So the format strings in ``pipeline.py`` are
an operational interface: a rename, a reorder, or a key inserted into the middle
of one of them breaks a search that currently works, silently.

Stage 3 appended exactly one key - ``cycle_id=%d`` - to eleven lines.  This file
pins the rule mechanically rather than by comment, by asserting for each line
that its format string is its **historical form plus the new key and nothing
else**.  That single assertion covers all four forbidden edits at once: a rename
changes the prefix, a reorder changes the prefix, an insertion shortens it, and a
"while I'm here" extra field makes the suffix more than ``cycle_id``.

It also pins the lines that were deliberately *not* touched, because "append
where it helps" and "append everywhere" look identical a month later.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The one key Stage 3 added, and the only thing any line below is allowed to
#: have gained.
CYCLE_ID_SUFFIX = " cycle_id=%d"

#: Line -> its format string as it stood before Stage 3.
#:
#: Keyed by the *whole* historical string rather than by an anchor plus tokens:
#: a shorter key would let the rest of the line drift.  The values are the
#: contract - do not "tidy" the wrapping, the spacing or the ``n/a`` literals in
#: them, because the point is that they are reproduced byte-for-byte on the wire.
UNCHANGED_PREFIX = {
    "inference started": (
        "inference started: camera_ts=%s pose_ts=%s frame_age_ms=%.1f"
    ),
    "takeover decision": (
        "takeover decision: control_flag=%d mean_lat_error=%s max_lat_error=%s "
        "takeover_count=%d release_count=%d reason=%s delay_s=%s start_index=%s "
        "remaining_points=%d reference_source=%s reference_fallback_reason=%s"
    ),
    "inference sent": (
        "inference sent: control_flag=%d waypoints=%d waypoint_seq=%d "
        "frame_id=%s camera_ts=%s pose_ts=%s frame_age_ms=%.1f "
        "camera_pose_delta_ms=%.1f inference_latency_ms=%.1f "
        "first=(%.7f,%.7f) last=(%.7f,%.7f)"
    ),
    "waypoint trim": (
        "waypoint trim: original_waypoint_count=%d matched_segment_index=%d "
        "projection_distance_m=%.3f removed_waypoint_count=%d "
        "remaining_waypoint_count=%d latest_pose_timestamp=%s "
        "trajectory_timestamp=%s"
    ),
    "waypoint trim skipped": (
        "waypoint trim skipped: reason=%s original_waypoint_count=%d "
        "matched_segment_index=n/a projection_distance_m=n/a "
        "removed_waypoint_count=0 remaining_waypoint_count=%d "
        "latest_pose_timestamp=%s trajectory_timestamp=%s"
    ),
    "trajectory expired": (
        "trajectory_expired / no_remaining_waypoints: "
        "original_waypoint_count=%d matched_segment_index=%s "
        "projection_distance_m=%s removed_waypoint_count=%d "
        "remaining_waypoint_count=0 latest_pose_timestamp=%s "
        "trajectory_timestamp=%s; sending count=0 control=0"
    ),
    "freshness gate blocked": (
        "freshness gate blocked inference: reason=%s frame_id=%s camera_ts=%s "
        "pose_ts=%s frame_age_ms=%.1f pose_age_ms=%.1f delta_ms=%.1f"
    ),
    "camera window stale": (
        "camera window is stale: t0=%s is %.0f ms behind latest pose ts=%s; "
        "skipping inference (if the lag is implausibly large, camera and pose "
        "timestamps may use different time domains)"
    ),
    "invalid trajectory": "invalid trajectory from model",
    "pipeline cycle failed": "pipeline cycle failed",
}

#: Event lines introduced with the multi-GPU pool, listed in full.
#:
#: These are the multi-process counterparts of ``inference started`` and
#: ``pipeline cycle failed``: one line per frozen input, carrying ``cycle_id``
#: because it is the same counter - only a complete, accepted input consumes
#: one, in either mode.
#:
#: Deliberately a separate map rather than more entries in ``UNCHANGED_PREFIX``.
#: That map means "the historical string plus the appended key, byte for byte",
#: which is the whole point of its assertion; a line that never had a historical
#: form would put a string in it that no edit can be checked against, and the
#: next person to add a line would follow that precedent.  A new line is new,
#: and says so here.
NEW_EVENT_LINES = {
    "task submitted": (
        "task submitted: task_seq=%d capture_ts=%s cycle_id=%d verdict=%s "
        "ready_workers=%d busy_workers=%d pending=%d"
    ),
    "model stop hold": (
        "model stop hold: control_flag=%d waypoints=%d "
        "latest_pose_timestamp=%s trajectory_timestamp=%s cycle_id=%d"
    ),
    "model stop has no current pose": (
        "model stop has no current pose to hold; not sending cycle_id=%d"
    ),
}

#: Lines that must NOT carry the new key, each with the reason it is excluded.
#: ``infer-package-saved`` ends in a path and ``no camera window`` in a JSON
#: blob, so a trailing key would sit inside a value that is already parsed.
UNTOUCHED = {
    "infer-package-saved": "the tail is a filesystem path",
    "no camera window": "the tail is a JSON blob (camera stats)",
}

#: A printf conversion, including the ``%%`` escape - which is not an argument.
_SPECIFIER = re.compile(
    r"%(?:[-+ #0]*)(?:\*|\d+)?(?:\.(?:\*|\d+))?[hlL]?([diouxXeEfFgGcrsa%])"
)


def _format_strings() -> dict[str, str]:
    """Every ``logger.*`` format string in pipeline.py, keyed by its text."""
    tree = ast.parse((ROOT / "pipeline.py").read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not ast.unparse(node.func).startswith("logger."):
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        if "%" not in first.value:
            continue
        out[first.value] = str(first.value)
    return out


def _arg_counts() -> dict[str, int]:
    tree = ast.parse((ROOT / "pipeline.py").read_text(encoding="utf-8"))
    out: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not ast.unparse(node.func).startswith("logger."):
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        if "%" not in first.value:
            continue
        out[first.value] = len(node.args) - 1
    return out


def _specifier_count(fmt: str) -> int:
    return sum(1 for m in _SPECIFIER.finditer(fmt) if m.group(1) != "%")


class AppendedKeyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strings = _format_strings()
        self.by_prefix = {}
        for fmt in self.strings:
            for label, prefix in UNCHANGED_PREFIX.items():
                if fmt.startswith(prefix):
                    self.by_prefix[label] = fmt

    def test_every_tracked_line_is_present(self):
        missing = sorted(set(UNCHANGED_PREFIX) - set(self.by_prefix))
        self.assertEqual(missing, [], f"these log lines disappeared: {missing}")

    def test_each_line_is_its_historical_form_plus_the_new_key(self):
        for label, prefix in UNCHANGED_PREFIX.items():
            with self.subTest(line=label):
                fmt = self.by_prefix.get(label)
                self.assertIsNotNone(fmt, f"{label} is gone")
                self.assertEqual(
                    fmt,
                    prefix + CYCLE_ID_SUFFIX,
                    "the historical part of this format string changed.  It is "
                    "grepped in the field: append the new key at the end, never "
                    "rename, reorder or insert into the middle.",
                )

    def test_the_appended_key_is_a_conversion_not_a_literal(self):
        # ``cycle_id=5`` hardcoded would satisfy the shape check above and log
        # the same number forever.
        for label, fmt in self.by_prefix.items():
            with self.subTest(line=label):
                self.assertTrue(
                    fmt.endswith(CYCLE_ID_SUFFIX),
                    f"{label} must end with {CYCLE_ID_SUFFIX!r}",
                )

    def test_every_new_event_line_is_present(self):
        # Otherwise NEW_EVENT_LINES would be a list of strings that matches
        # nothing, and test_no_other_line_gained_a_key would pass vacuously by
        # exempting a line that no longer exists.
        for label, fmt in NEW_EVENT_LINES.items():
            with self.subTest(line=label):
                self.assertIn(fmt, self.strings, f"the {label} line is gone")

    def test_no_other_line_gained_a_key(self):
        tracked = set(self.by_prefix.values()) | set(NEW_EVENT_LINES.values())
        for label, fmt in self.strings.items():
            if fmt in tracked:
                continue
            with self.subTest(fmt=label):
                self.assertNotIn(
                    "cycle_id",
                    fmt,
                    "cycle_id goes on the event lines that describe a loop "
                    "iteration, not everywhere a logger is called.  A genuinely "
                    "new event line belongs in NEW_EVENT_LINES, where the "
                    "decision to carry it is visible.",
                )

    def test_the_untouched_lines_are_still_untouched(self):
        for label, prefix in UNTOUCHED.items():
            with self.subTest(line=label):
                matching = [f for f in self.strings if label in f]
                self.assertTrue(matching, f"{label} is gone")
                for fmt in matching:
                    self.assertNotIn(
                        "cycle_id",
                        fmt,
                        f"{label}'s tail is parsed ({prefix}); a key appended "
                        "after it lands inside that value",
                    )


class FormatArgumentBalanceTests(unittest.TestCase):
    """Every tracked line passes exactly as many arguments as it formats.

    A missing argument is a ``TypeError`` at log time - which, on a line like
    ``pipeline cycle failed``, means the error that was being reported is
    replaced by an error about reporting it.  None of these paths can be
    exercised locally (they need a camera, a model, or a fault), so the balance
    is checked statically instead.
    """

    def test_specifier_count_matches_argument_count(self):
        counts = _arg_counts()
        tracked = {
            fmt for fmt in counts
            for prefix in UNCHANGED_PREFIX.values()
            if fmt.startswith(prefix)
        }
        self.assertEqual(
            len(tracked), len(UNCHANGED_PREFIX),
            "not every tracked line was found",
        )
        for fmt in sorted(tracked):
            with self.subTest(fmt=fmt):
                self.assertEqual(
                    _specifier_count(fmt),
                    counts[fmt],
                    f"{counts[fmt]} arguments for {_specifier_count(fmt)} "
                    f"conversions in {fmt!r}",
                )

    def test_the_checker_counts_specifiers_correctly(self):
        # Without this, a broken regex would make the test above pass vacuously.
        self.assertEqual(_specifier_count("a=%d b=%s c=%.3f"), 3)
        self.assertEqual(_specifier_count("%.7f"), 1)
        self.assertEqual(_specifier_count("100%% done"), 0)
        self.assertEqual(_specifier_count("%d and 50%% of %s"), 2)


class FieldSynonymTests(unittest.TestCase):
    """The two values that are logged under two names each stay documented.

    ``camera_ts`` / ``trajectory_timestamp`` are both ``result.t0_us``, and
    ``pose_ts`` / ``latest_pose_timestamp`` are both the pose the crop consumed.
    Renaming either spelling to match the other would break a field grep, so
    the mapping has to be written down where the next reader will find it.
    """

    def test_the_mapping_is_recorded_in_pipeline(self):
        source = (ROOT / "pipeline.py").read_text(encoding="utf-8")
        self.assertIn("Log field synonyms", source)
        for name in (
            "camera_ts",
            "trajectory_timestamp",
            "pose_ts",
            "latest_pose_timestamp",
        ):
            with self.subTest(name=name):
                self.assertIn(name, source)

    def test_both_spellings_are_actually_still_in_use(self):
        strings = _format_strings()
        send = next(f for f in strings if f.startswith("inference sent"))
        trim = next(f for f in strings if f.startswith("waypoint trim:"))
        self.assertIn("camera_ts=%s", send)
        self.assertIn("pose_ts=%s", send)
        self.assertIn("trajectory_timestamp=%s", trim)
        self.assertIn("latest_pose_timestamp=%s", trim)


if __name__ == "__main__":
    unittest.main()
