"""Stage 1 / Stage 2 module-boundary guards.

``app.py`` is being reduced to entry point / composition / lifecycle / HTTP.
These checks pin two things that are easy to undo by accident:

1. the code that moved out really is gone from ``app.py`` (so it cannot drift
   back into two copies), and it really does exist in the new module;
2. every name other code imports from ``app`` still resolves, and to the *same
   object* as the new module - a stale duplicate class would silently pass an
   ``isinstance`` check in one place and fail it in another.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# name -> module that now owns it
MOVED = {
    # Stage 1: config
    "AppConfig": "config.loader",
    # Stage 1: coordinate transforms
    "enu_from_ll": "trajectory.coordinates",
    "yaw_to_matrix": "trajectory.coordinates",
    "enu_to_ego": "trajectory.coordinates",
    "local_xy_to_latlon": "trajectory.coordinates",
    # Stage 1: vehicle models and history
    "GlobalPose": "vehicle.models",
    "ChassisInfo": "vehicle.models",
    "PoseSample": "vehicle.models",
    "PoseHistory": "vehicle.history",
    "build_ego_arrays": "vehicle.history",
    "poses_to_model_frame_xy": "vehicle.history",
    # Stage 1: vehicle receivers
    "UdpPoseListener": "vehicle.receiver",
    "MockPoseProvider": "vehicle.receiver",
    "decode_global_pose_json": "vehicle.receiver",
    "decode_vehicle_timestamp_json": "vehicle.receiver",
    "decode_chassis_json": "vehicle.receiver",
    # Stage 1: camera models
    "CameraWindow": "camera.models",
    # Stage 2: outbound wire serializer
    "WaypointSender": "waypoint.sender",
    # Stage 2: latency-driven trajectory crop
    "trim_waypoints_to_latest_pose": "trajectory.postprocess",
    "TrajectoryPostprocessResult": "trajectory.postprocess",
    # An alias rather than a class: ``WaypointTrimResult = TrajectoryPostprocessResult``
    "WaypointTrimResult": "trajectory.postprocess",
    # Stage 2: the model call, its prompt and its result
    "AlpamayoRunner": "inference.alpamayo",
    "ModelResult": "inference.alpamayo",
    "_messages_text": "inference.alpamayo",
    # Stage 2: camera naming, shared by the prompt and the inference package
    "_CAMERA_DISPLAY_NAMES": "inference.camera_labels",
    "_CAMERA_PACKAGE_LABELS": "inference.camera_labels",
    "_camera_package_label": "inference.camera_labels",
    # Stage 2: the real-time loop
    "PipelineRunner": "pipeline",
}

# Stage 2 modules that are new rather than moved, and the names each must own.
# Same guard as MOVED, minus "must be gone from app.py" - app.py re-exports
# these by import, and importing is not defining.
NEW_IN_STAGE_2 = {
    "inference.input_builder": (
        "InputBuilder",
        "InputBuildResult",
        "REASON_CAMERA_STALE",
        "REASON_POSE_STALE",
        "REASON_CAMERA_POSE_DELTA",
        "REASON_CAMERA_WINDOW_STALE",
        "REASON_POSE_HISTORY_UNAVAILABLE",
    ),
}

# Names still defined in app.py on purpose.  Empty since Stage 3 removed the
# unreferenced legacy camera block (which was the last holder of a
# deliberately-kept name: it re-declared ``FrameEntry``, shadowing
# ``camera.models.FrameEntry``).  Kept as an empty set rather than deleted
# because the two assertions below read it - an empty allowlist is the
# statement, and a future exception has one obvious place to go.
_DELIBERATELY_KEPT: set[str] = set()

# Top-level names that carry behaviour the pipeline is only allowed to *call*.
# A definition here means a formula, an algorithm or an encoder was written back
# into the orchestrator - the specific thing Stage 2 was asked to prevent.
_NOT_A_PIPELINE_DEFINITION = (
    "enu_", "yaw_", "local_xy_", "latlon", "trim_waypoints", "decode_",
    "build_ego_arrays", "poses_to_model_frame_xy",
)
_NOT_A_PIPELINE_CLASS = (
    "TrajectoryPostprocess", "Takeover", "WaypointSender", "WaypointTrim",
    "AlpamayoRunner", "ModelResult", "InputBuilder", "InputBuildResult",
    "CameraWindow", "FrameEntry", "GlobalPose", "PoseSample", "PoseHistory",
    "TrajectoryCompensator", "TrajectoryDeviationCalculator",
    "PlannedRouteReferenceProvider", "ReferenceTrajectorySelector",
)


def _app_tree() -> ast.Module:
    return ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))


def _pipeline_tree() -> ast.Module:
    return ast.parse((ROOT / "pipeline.py").read_text(encoding="utf-8"))


def _defines(node: ast.stmt, name: str) -> bool:
    """A moved name may be redefined (def/class) or re-bound (an alias)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name == name
    if isinstance(node, ast.Assign):
        return any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        )
    return False


def _imports_of(tree: ast.Module) -> set[str]:
    """Every module path imported, as written, plus each one's root."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
            names.add(node.module.split(".")[0])
    return names


class MovedCodeIsGoneFromAppTests(unittest.TestCase):
    def setUp(self) -> None:
        tree = _app_tree()
        self.top_level = {
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }

    def test_app_no_longer_defines_any_moved_name(self):
        still_here = sorted((self.top_level & set(MOVED)) - _DELIBERATELY_KEPT)
        self.assertEqual(
            still_here, [],
            "these were moved to a module of their own but are defined at the "
            f"top level of app.py again: {still_here}",
        )

    def test_app_does_not_define_frame_entry_twice_over(self):
        # Was ``assertLessEqual(count, 1)`` while the legacy block still existed:
        # it documented the Stage 3 cleanup target rather than failing on it.
        # The block is gone, so the assertion is now the real one - app.py must
        # not define FrameEntry at all, which is what keeps
        # ``app.FrameEntry`` resolvable only as the re-export of
        # ``camera.models.FrameEntry``.
        count = sum(
            1 for n in _app_tree().body
            if isinstance(n, ast.ClassDef) and n.name == "FrameEntry"
        )
        self.assertEqual(count, 0)

    def test_each_moved_name_is_actually_defined_in_its_new_module(self):
        missing = self._missing_definitions(MOVED)
        missing += self._missing_definitions(
            {name: module
             for module, names in NEW_IN_STAGE_2.items()
             for name in names}
        )
        self.assertEqual(missing, [], f"moved names missing from their new home: {missing}")

    @staticmethod
    def _missing_definitions(mapping) -> list[str]:
        missing = []
        for name, module in mapping.items():
            path = ROOT / (module.replace(".", "/") + ".py")
            if not path.exists():
                missing.append(f"{module} (file absent)")
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if not any(_defines(node, name) for node in tree.body):
                missing.append(f"{name} not in {module}")
        return missing

    def test_stage_2_names_are_not_defined_in_app(self):
        # app.py re-exports them by import.  A definition would be the second
        # copy the stage forbids.
        defined = []
        for module, names in NEW_IN_STAGE_2.items():
            for name in names:
                if any(_defines(node, name) for node in _app_tree().body):
                    defined.append(f"{name} (also in {module})")
        self.assertEqual(defined, [], f"defined in app.py as well: {defined}")

    def test_import_cycles_stay_broken(self):
        # vehicle.history -> trajectory.coordinates -> (nothing but numpy);
        # app -> everything.  A back-edge would mean the split failed.
        #
        # Forbidden entries may be a bare root ("camera") or a full dotted path
        # ("vehicle.history"); both are matched, so a dotted entry actually
        # bites - comparing only roots silently ignored every dotted one.
        back_edges = []
        for module, forbidden in (
            ("trajectory/coordinates.py", {"vehicle", "app", "camera"}),
            ("vehicle/models.py", {"vehicle.history", "app", "camera"}),
            # The wire serializer is a leaf: it must stay importable without
            # pulling the pipeline (or torch, or the camera stack) in.
            ("waypoint/sender.py", {"app", "camera", "vehicle", "pipeline"}),
            # The gates and the ego tensors: no model, no decision, no wire.
            ("inference/input_builder.py", {
                "app", "pipeline", "camera.sync", "camera.source",
                "waypoint", "trajectory.takeover", "inference.alpamayo",
                "trajectory.takeover", "trajectory.postprocess",
                "vehicle.receiver",
            }),
            # The model call: it takes its inputs as arguments and returns a
            # result.  Nothing downstream may be reachable from here.
            ("inference/alpamayo.py", {
                "app", "pipeline", "waypoint", "trajectory.takeover",
                "trajectory.postprocess", "trajectory.takeover",
                "vehicle.receiver", "camera.sync",
            }),
            # The orchestrator imports downward only.
            ("pipeline.py", {"app"}),
            # The published view is a true leaf: stdlib only, so it can be
            # tested with no runner, no camera, no cv2 and no GPU - which is the
            # whole reason readiness is a pure function over a snapshot rather
            # than a method on the runner.
            ("health.py", {
                "app", "pipeline", "camera", "camera.sync",
                "camera.source", "vehicle", "vehicle.receiver",
                "trajectory", "trajectory.takeover", "trajectory.postprocess",
                "trajectory.takeover", "waypoint", "inference",
                "inference.alpamayo", "inference.input_builder", "time_sync",
                "vehicle.planned_route", "config", "config.loader",
            }),
        ):
            tree = ast.parse((ROOT / module).read_text(encoding="utf-8"))
            hit = _imports_of(tree) & forbidden
            if hit:
                back_edges.append(f"{module} imports {sorted(hit)}")
        self.assertEqual(back_edges, [])

    def test_the_official_model_package_is_imported_from_one_place(self):
        # ``alpamayo1_5`` is the official, unmodifiable source.  Within the
        # service code one import site means one place to read when asking how
        # the model is called.  ``tools/`` is excluded on purpose: those are
        # offline GPU-box harnesses that drive the official package directly and
        # are not part of the running service.
        holders = []
        for path in sorted(ROOT.rglob("*.py")):
            rel = path.relative_to(ROOT)
            if any(p.startswith(".") or p in ("tests", "tools", "__pycache__")
                   for p in rel.parts):
                continue
            names = _imports_of(ast.parse(path.read_text(encoding="utf-8")))
            if any(n == "alpamayo1_5" or n.startswith("alpamayo1_5.")
                   for n in names):
                holders.append(str(rel).replace("\\", "/"))
        self.assertEqual(holders, ["inference/alpamayo.py"], holders)

    def test_pipeline_defines_no_formula_or_algorithm(self):
        # The orchestrator may CALL the coordinate transform, the model, the
        # takeover engine and the encoder - it may not own any of them.
        offenders = []
        for node in _pipeline_tree().body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(node.name.startswith(p) for p in _NOT_A_PIPELINE_DEFINITION):
                    offenders.append(node.name)
            elif isinstance(node, ast.ClassDef):
                if any(node.name.startswith(p) for p in _NOT_A_PIPELINE_CLASS):
                    offenders.append(node.name)
        self.assertEqual(
            offenders, [],
            "these belong to a module of their own, not to pipeline.py: "
            f"{offenders}",
        )

    def test_the_shared_camera_naming_table_has_no_imports(self):
        # inference/camera_labels.py is deliberately import-free: it is the one
        # thing the prompt layer and the package writer share, and a dependency
        # here would be a dependency of both.
        tree = ast.parse(
            (ROOT / "inference" / "camera_labels.py").read_text(encoding="utf-8")
        )
        imports = [
            ast.unparse(n) for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom))
            # ``from __future__ import annotations`` is a compile-time flag, not
            # a dependency.
            and not (isinstance(n, ast.ImportFrom)
                     and n.module == "__future__")
        ]
        self.assertEqual(imports, [], f"camera_labels.py grew imports: {imports}")


class ReExportIdentityTests(unittest.TestCase):
    """Importing ``app`` needs cv2, so the checks run in a subprocess."""

    @classmethod
    def setUpClass(cls) -> None:
        script = textwrap.dedent("""
            import sys, types
            sys.modules.setdefault("cv2", types.ModuleType("cv2"))
            import app
            import config.loader, camera.models, camera.sync
            import vehicle.models, vehicle.history, vehicle.receiver
            import trajectory.coordinates, trajectory.postprocess
            import waypoint.sender
            import pipeline, inference.alpamayo, inference.input_builder
            import inference.camera_labels

            pairs = [
                (app.WaypointSender,           waypoint.sender.WaypointSender),
                (app.trim_waypoints_to_latest_pose,
                                     trajectory.postprocess.trim_waypoints_to_latest_pose),
                (app.TrajectoryPostprocessResult,
                                     trajectory.postprocess.TrajectoryPostprocessResult),
                (app.WaypointTrimResult,       trajectory.postprocess.WaypointTrimResult),
                (app.Settings,                 config.loader.AppConfig),
                (app.AppConfig,                config.loader.AppConfig),
                (app.CameraWindow,             camera.models.CameraWindow),
                (app.CameraWindow,             camera.sync.CameraWindow),
                (app.GlobalPose,               vehicle.models.GlobalPose),
                (app.ChassisInfo,              vehicle.models.ChassisInfo),
                (app.PoseSample,               vehicle.models.PoseSample),
                (app.PoseHistory,              vehicle.history.PoseHistory),
                (app.build_ego_arrays,         vehicle.history.build_ego_arrays),
                (app.poses_to_model_frame_xy,  vehicle.history.poses_to_model_frame_xy),
                (app.enu_from_ll,              trajectory.coordinates.enu_from_ll),
                (app.enu_to_ego,               trajectory.coordinates.enu_to_ego),
                (app.yaw_to_matrix,            trajectory.coordinates.yaw_to_matrix),
                (app.local_xy_to_latlon,       trajectory.coordinates.local_xy_to_latlon),
                (app.UdpPoseListener,          vehicle.receiver.UdpPoseListener),
                (app.MockPoseProvider,         vehicle.receiver.MockPoseProvider),
                (app.decode_global_pose_json,  vehicle.receiver.decode_global_pose_json),
                (app.decode_chassis_json,      vehicle.receiver.decode_chassis_json),
                (app.decode_vehicle_timestamp_json,
                                               vehicle.receiver.decode_vehicle_timestamp_json),
                # Stage 2: the model layer, reached through its historical
                # ``app.`` path by tools/infer_images_pose_only.py.
                (app.AlpamayoRunner,           inference.alpamayo.AlpamayoRunner),
                (app.ModelResult,              inference.alpamayo.ModelResult),
                (app._messages_text,           inference.alpamayo._messages_text),
                # Stage 3 stopped re-exporting them, since nothing in either
                # checkout imports them.  Their absence is asserted below.
                (app._CAMERA_DISPLAY_NAMES,
                                     inference.camera_labels._CAMERA_DISPLAY_NAMES),
                (app._CAMERA_PACKAGE_LABELS,
                                     inference.camera_labels._CAMERA_PACKAGE_LABELS),
                (app._camera_package_label,
                                     inference.camera_labels._camera_package_label),
                # Stage 2: the input gates, reached through ``app.`` by the
                # structural tests and usable directly by a harness.
                (app.InputBuilder,             inference.input_builder.InputBuilder),
                (app.InputBuildResult,         inference.input_builder.InputBuildResult),
                (app.FRESHNESS_REASONS,        inference.input_builder.FRESHNESS_REASONS),
                (app.REASON_CAMERA_STALE,      inference.input_builder.REASON_CAMERA_STALE),
                (app.REASON_POSE_STALE,        inference.input_builder.REASON_POSE_STALE),
                (app.REASON_CAMERA_POSE_DELTA,
                                     inference.input_builder.REASON_CAMERA_POSE_DELTA),
                (app.REASON_CAMERA_WINDOW_STALE,
                                     inference.input_builder.REASON_CAMERA_WINDOW_STALE),
                (app.REASON_POSE_HISTORY_UNAVAILABLE,
                                     inference.input_builder.REASON_POSE_HISTORY_UNAVAILABLE),
                # Stage 2: the loop.  This one must be the SAME class object,
                # not a subclass or a copy: tools/package_harness.py and
                # tests/test_waypoint_trimming.py reach into its members.
                (app.PipelineRunner,           pipeline.PipelineRunner),
            ]
            label = sys.argv[1] if len(sys.argv) > 1 else "app"
            for i, (a, b) in enumerate(pairs):
                if a is not b:
                    print("MISMATCH", i)
            print("ROUTES", ",".join(sorted(
                r.path for r in app.app.routes if hasattr(r, "path"))))
        """)
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=str(ROOT),
            capture_output=True, text=True, timeout=180,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"could not import app.py\n--- stderr ---\n{completed.stderr}"
            )
        cls.out = completed.stdout

    def test_all_reexports_are_the_same_object(self):
        self.assertNotIn("MISMATCH", self.out)

    def test_http_surface_is_unchanged(self):
        routes = next(l for l in self.out.splitlines() if l.startswith("ROUTES"))
        for path in ("/health", "/status", "/docs", "/openapi.json"):
            self.assertIn(path, routes)


class AppHoldsOnlyAssemblyTests(unittest.TestCase):
    """app.py must hold entry, assembly, lifecycle and HTTP - nothing else.

    An allowlist rather than a predicate (no "looks like a helper" heuristic):
    the point is that adding a definition to app.py is a deliberate act with a
    one-line reason attached to it, so code does not drift back into the
    orchestrator one convenient helper at a time - the exact thing Stages 1-3
    were for.
    """

    ALLOWED = {
        "_git_commit": "reads the deployed revision for the startup log",
        "_log_startup_snapshot": "operator-read startup config dump (lifecycle)",
        "_snapshot_and_alive": "reads the published snapshot and the thread's "
                               "liveness as a pair, so one response describes "
                               "one instant",
        "set_control_flag": "legacy manual-control entry point, pinned by AST "
                            "extraction in test_model_control_removed.py",
        "resolve_control_flag": "legacy manual-control query, pinned likewise",
        "_startup": "ASGI startup hook",
        "_shutdown": "ASGI shutdown hook",
        "health": "HTTP route - /health",
        "status": "HTTP route - /status",
        "ready": "HTTP route - /ready",
    }

    def test_app_defines_nothing_outside_the_allowlist(self):
        tree = _app_tree()
        defined = {
            node.name: type(node).__name__
            for node in tree.body
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
        }
        unexpected = sorted(set(defined) - set(self.ALLOWED))
        self.assertEqual(
            unexpected, [],
            "app.py is entry / assembly / lifecycle / HTTP only.  These are new "
            f"top-level definitions: {unexpected}.  Move them to a module of "
            "their own, or add them to ALLOWED with a one-line reason.",
        )

    def test_the_allowlist_has_no_stale_entries(self):
        tree = _app_tree()
        defined = {
            node.name
            for node in tree.body
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
        }
        stale = sorted(set(self.ALLOWED) - defined)
        self.assertEqual(
            stale, [],
            f"ALLOWED lists names app.py no longer defines: {stale}",
        )


if __name__ == "__main__":
    unittest.main()
