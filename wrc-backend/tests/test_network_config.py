"""The network section: defaults, precedence, validation and conflict rules.

The point of these cases is the wire contract, not the dataclass.  Every
listener and target the vehicle runs is built from ``load_network_config``, so
these tests pin down (a) that an untouched deployment still gets exactly the
addresses and ports it has always used, (b) that a document or an environment
variable can move one of them, and (c) that a configuration which cannot work -
a port out of range, a wildcard where a target belongs, two listeners one
process cannot both bind - is rejected with the endpoint named, before any
socket is opened.
"""

from __future__ import annotations

import inspect
import io
import json
import os
import re
import socket
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest.mock import mock_open, patch

from _cv2_isolation import cv2_stub
from config.loader import (
    NetworkConfig,
    load_network_config,
    load_backend_config,
)

# Direct PGX endpoints.  A change here moves a socket the vehicle uses.
DIRECT_DEFAULTS = {
    "cimg": ("0.0.0.0", 30106, "tcp", "listen"),
    "pose_udp": ("0.0.0.0", 7001, "udp", "listen"),
    "waypoint": ("168.130.182.10", 7010, "udp", "remote"),
    "route_tcp": ("0.0.0.0", 30006, "tcp", "listen"),
    "http": ("0.0.0.0", 8090, "tcp", "listen"),
}

# Every ALPAMAYO_* variable this module may consult, cleared per test so the
# developer's own shell cannot decide the outcome.
_MANAGED = (
    "ALPAMAYO_CONFIG",
    "ALPAMAYO_CIMG_HOST",
    "ALPAMAYO_CIMG_PORT",
    "ALPAMAYO_POSE_UDP_HOST",
    "ALPAMAYO_POSE_UDP_PORT",
    "ALPAMAYO_WAYPOINT_HOST",
    "ALPAMAYO_WAYPOINT_PORT",
    "ALPAMAYO_ROUTE_TCP_HOST",
    "ALPAMAYO_ROUTE_TCP_PORT",
    "ALPAMAYO_ROUTE_TCP_ENABLED",
    "ALPAMAYO_HOST",
    "ALPAMAYO_PORT",
    "ALPAMAYO_RELAY_TCP_HOST",
    "ALPAMAYO_RELAY_TCP_PORT",
    "ALPAMAYO_RELAY_INJECT_HOST",
    "ALPAMAYO_RELAY_INJECT_PORT",
    "ALPAMAYO_RELAY_DOWNLINK_HOST",
    "ALPAMAYO_RELAY_DOWNLINK_PORT",
    "HF_ENDPOINT",
    "ALPAMAYO_PACKAGE_TEST_URL",
    "ALPAMAYO_CAMERA_INPUT",
    "ALPAMAYO_CAMERA_SOURCES",
)

# The document that ships with the repository, which the launcher reads.
SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.json"


class _CleanEnv(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in _MANAGED:
            self.addCleanup(os.environ.pop, name, None)
            os.environ.pop(name, None)

    def write_document(self, network: dict, **extra: dict) -> str:
        """A config.json holding ``network``, in a temporary directory."""
        payload = {"network": network, **extra}
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        with handle:
            json.dump(payload, handle)
        self.addCleanup(os.unlink, handle.name)
        return handle.name


class DefaultValueTests(_CleanEnv):
    def test_every_default_is_the_direct_pgx_endpoint(self) -> None:
        config = load_network_config()
        for role, (host, port, protocol, direction) in DIRECT_DEFAULTS.items():
            endpoint = getattr(config, role.replace(".", "_"))
            with self.subTest(role=role):
                self.assertEqual(host, endpoint.host)
                self.assertEqual(port, endpoint.port)
                self.assertEqual(protocol, endpoint.protocol)
                self.assertEqual(direction, endpoint.direction)

    def test_the_shipped_document_agrees_with_the_defaults(self) -> None:
        """config/config.json repeats the defaults, so adding it changed nothing.

        This is what makes the file safe to ship: a deployment that keeps using
        it starts the pipeline byte-identically to the build before the section
        existed.
        """
        document = json.loads(SHIPPED_CONFIG.read_text(encoding="utf-8"))
        network = document["network"]
        self.assertEqual(network["cimg"], {"host": "0.0.0.0", "port": 30106})
        self.assertEqual(network["pose_udp"], {"host": "0.0.0.0", "port": 7001})
        self.assertEqual(
            network["waypoint"], {"host": "168.130.182.10", "port": 7010}
        )
        self.assertEqual(
            network["route_tcp"], {"enabled": True, "host": "0.0.0.0", "port": 30006}
        )
        self.assertEqual(network["http"], {"host": "0.0.0.0", "port": 8090})
        self.assertNotIn("relay", network)
        self.assertEqual(
            network["http_client"],
            {
                "hf_endpoint": "https://hf-mirror.com",
                "package_test_url": "http://127.0.0.1:8091/infer",
            },
        )
        self.assertNotIn("camera_sources", network)

    def test_the_http_client_urls_are_the_ones_that_were_written_down(self) -> None:
        """``hf_endpoint`` was exported by the launcher and the offline test URL
        was a literal in ``tools/request_model_test.py``; both are now fields,
        with the same two values."""
        config = load_network_config()
        self.assertEqual("https://hf-mirror.com", config.hf_endpoint)
        self.assertEqual("http://127.0.0.1:8091/infer", config.package_test_url)

    def test_a_named_document_that_is_missing_is_refused(self) -> None:
        """A file asked for by name never becomes the built-in defaults.

        This is the failure worth pinning down: the operator edits a document,
        the path to it is wrong, and the vehicle starts on the addresses it used
        before without a word.  A named document - ``--config`` or
        ``ALPAMAYO_CONFIG`` - is a request, and a request that cannot be met is
        an error.
        """
        missing = SHIPPED_CONFIG.parent / "does-not-exist.json"
        with self.assertRaises(ValueError) as caught:
            load_network_config(str(missing))
        self.assertIn("--config", str(caught.exception))

        os.environ["ALPAMAYO_CONFIG"] = str(missing)
        with self.assertRaises(ValueError) as caught:
            load_network_config()
        self.assertIn("ALPAMAYO_CONFIG", str(caught.exception))

    def test_a_document_that_is_present_but_unparseable_is_refused(self) -> None:
        broken = Path(self.write_document({})).with_suffix(".broken")
        broken.write_text("{not json", encoding="utf-8")
        self.addCleanup(broken.unlink)
        with self.assertRaises(ValueError) as caught:
            load_network_config(str(broken))
        self.assertIn(str(broken), str(caught.exception))

    def test_the_shipped_document_is_required_to_be_readable(self) -> None:
        """Present but corrupt is a mistake; absent is a legacy deployment."""
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "config.json"
            broken.write_text("{not json", encoding="utf-8")
            with patch("config.loader.resolve_config_path", return_value=broken):
                with self.assertRaises(ValueError):
                    load_network_config()
        with patch("config.loader.resolve_config_path", return_value=None):
            config = load_network_config()
        self.assertEqual(config.http.port, 8090)
        self.assertEqual(config.cimg.port, 30106)


class PrecedenceTests(_CleanEnv):
    def test_the_document_moves_an_endpoint(self) -> None:
        path = self.write_document(
            {"cimg": {"host": "127.0.0.1", "port": 39106}, "http": {"port": 18090}}
        )
        config = load_network_config(path)
        self.assertEqual(("127.0.0.1", 39106), (config.cimg.host, config.cimg.port))
        self.assertEqual(18090, config.http.port)
        # Everything not named keeps its default.
        self.assertEqual(7001, config.pose_udp.port)

    def test_the_environment_overrides_the_document(self) -> None:
        path = self.write_document({"cimg": {"host": "127.0.0.1", "port": 39106}})
        os.environ["ALPAMAYO_CIMG_PORT"] = "39999"
        config = load_network_config(path)
        self.assertEqual(39999, config.cimg.port)
        # The host was not overridden, so it still comes from the document.
        self.assertEqual("127.0.0.1", config.cimg.host)

    def test_an_empty_environment_variable_counts_as_unset(self) -> None:
        os.environ["ALPAMAYO_CIMG_HOST"] = ""
        config = load_network_config()
        self.assertEqual("0.0.0.0", config.cimg.host)

    def test_route_tcp_enabled_follows_the_document_and_the_environment(self) -> None:
        path = self.write_document({"route_tcp": {"enabled": False}})
        self.assertFalse(load_network_config(path).route_tcp_enabled)
        os.environ["ALPAMAYO_ROUTE_TCP_ENABLED"] = "true"
        self.assertTrue(load_network_config(path).route_tcp_enabled)

    def test_the_shipped_config_path_is_used_when_none_is_given(self) -> None:
        """``ALPAMAYO_CONFIG`` selects the document ``load_app_config`` reads."""
        os.environ["ALPAMAYO_CONFIG"] = self.write_document({"cimg": {"port": 30107}})
        self.assertEqual(30107, load_network_config().cimg.port)
        self.assertEqual(30107, load_backend_config()["network"]["cimg"]["port"])


class ValidationTests(_CleanEnv):
    def assert_rejected(self, network: dict, needle: str) -> str:
        path = self.write_document(network)
        with self.assertRaises(ValueError) as caught:
            load_network_config(path)
        message = str(caught.exception)
        self.assertIn(needle, message)
        return message

    def test_a_port_out_of_range_is_rejected_by_name(self) -> None:
        for port in (0, 70000, -1):
            with self.subTest(port=port):
                self.assert_rejected(
                    {"cimg": {"port": port}}, "network.cimg.port"
                )

    def test_a_non_integer_port_is_rejected(self) -> None:
        self.assert_rejected({"cimg": {"port": "eight"}}, "network.cimg.port")
        # A boolean where a port belongs is a typo, not the number one.
        self.assert_rejected({"cimg": {"port": True}}, "network.cimg.port")

    def test_a_digit_string_port_is_accepted(self) -> None:
        config = load_network_config(self.write_document({"cimg": {"port": "30108"}}))
        self.assertEqual(30108, config.cimg.port)

    def test_an_empty_host_is_rejected(self) -> None:
        self.assert_rejected({"cimg": {"host": ""}}, "network.cimg.host")

    def test_a_wildcard_where_a_target_belongs_is_rejected(self) -> None:
        for host in ("0.0.0.0", "::", "*"):
            with self.subTest(host=host):
                self.assert_rejected(
                    {"waypoint": {"host": host}}, "network.waypoint.host"
                )

    def test_a_wildcard_listener_is_fine(self) -> None:
        config = load_network_config(self.write_document({"cimg": {"host": "0.0.0.0"}}))
        self.assertEqual("0.0.0.0", config.cimg.host)

    def test_a_declared_protocol_or_direction_that_contradicts_the_role(self) -> None:
        self.assert_rejected({"cimg": {"protocol": "udp"}}, "not configurable")
        self.assert_rejected({"waypoint": {"direction": "listen"}}, "not configurable")

    def test_a_malformed_http_client_url_is_rejected(self) -> None:
        self.assert_rejected(
            {"http_client": {"hf_endpoint": "hf-mirror.com"}},
            "network.http_client.hf_endpoint",
        )

    def test_a_non_object_section_is_rejected(self) -> None:
        self.assert_rejected({"cimg": "0.0.0.0:30106"}, "network.cimg must be an object")

    def test_a_camera_sources_list_is_rejected(self) -> None:
        self.assert_rejected(
            {"camera_sources": ["udp://127.0.0.1:31101"]},
            "network.camera_sources",
        )


class RetiredCameraInputTests(_CleanEnv):
    """MPEG-TS inputs fail at load time instead of selecting a second camera path."""

    def test_a_camera_sources_document_is_rejected(self) -> None:
        path = self.write_document({"camera_sources": ["rtsp://cam1/stream"]})
        with self.assertRaises(ValueError) as caught:
            load_network_config(path)
        self.assertIn("network.camera_sources", str(caught.exception))

    def test_a_retired_camera_environment_variable_is_rejected(self) -> None:
        from config.loader import AppConfig

        os.environ["ALPAMAYO_CAMERA_SOURCES"] = "udp://127.0.0.1:31101"
        with self.assertRaises(ValueError) as caught:
            AppConfig.from_env()
        self.assertIn("ALPAMAYO_CAMERA_SOURCES", str(caught.exception))


class PoseListenerTests(_CleanEnv):
    """The pose endpoint reaches the socket, not only the startup log."""

    @staticmethod
    def _free_udp_port() -> int:
        """An OS-assigned port, so the case never sits on the vehicle's 7001."""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]

    def test_the_listener_binds_the_endpoint_it_is_given(self) -> None:
        from vehicle.history import PoseHistory
        from vehicle.receiver import UdpPoseListener

        port = self._free_udp_port()
        listener = UdpPoseListener(
            PoseHistory(), pose_host="127.0.0.1", pose_port=port
        )
        listener.start()
        self.addCleanup(listener.stop)
        # The listener binds on its thread, so the socket list fills in a moment.
        deadline = time.monotonic() + 5.0
        while not listener._sockets and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(listener._sockets, "the listener never bound a socket")
        self.assertEqual(
            ("127.0.0.1", port), listener._sockets[0].getsockname()[:2]
        )

    def test_the_pipeline_passes_the_configured_endpoint_to_the_listener(self) -> None:
        """``pose_host``/``pose_port`` are defaults on ``UdpPoseListener``, and
        ``PipelineRunner.start`` used not to pass the settings through - so a
        changed ``ALPAMAYO_POSE_UDP_PORT`` moved a log line and left the socket
        on 7001.  Read from the source rather than by starting the pipeline: a
        real start needs the model and the cameras, which is why the drift
        survived.  The socket itself is covered by the case above.
        """
        with cv2_stub():
            import pipeline

            source = inspect.getsource(pipeline.PipelineRunner.start)
        self.assertIn("pose_host=self.settings.pose_udp_host", source)
        self.assertIn("pose_port=self.settings.pose_udp_port", source)


class ListenConflictTests(_CleanEnv):
    def config_with(self, role: str, host: str, port: int) -> NetworkConfig:
        endpoint = replace(
            getattr(NetworkConfig(), role.replace(".", "_")), host=host, port=port
        )
        return replace(NetworkConfig(), **{role.replace(".", "_"): endpoint})

    def test_two_listeners_in_one_process_on_one_address_conflict(self) -> None:
        config = self.config_with("http", "127.0.0.1", 30106)
        # ``cimg`` already holds 0.0.0.0:30106 in the app process.
        conflicts = config.listen_conflicts()
        self.assertEqual(1, len(conflicts))
        self.assertIn("network.cimg", conflicts[0])
        self.assertIn("network.http", conflicts[0])

    def test_a_wildcard_overlaps_a_specific_address(self) -> None:
        config = self.config_with("http", "0.0.0.0", 30106)
        self.assertTrue(config.listen_conflicts())

    def test_a_different_address_on_one_port_is_not_a_conflict(self) -> None:
        """0.0.0.0:7001 and 127.0.0.1:7001 cannot both bind, but the two below
        are on addresses that do not overlap."""
        config = self.config_with("http", "127.0.0.1", 30106)
        config = replace(
            config,
            cimg=replace(config.cimg, host="192.168.1.10"),
        )
        self.assertEqual([], config.listen_conflicts())

    def test_tcp_and_udp_on_the_same_port_are_not_a_conflict(self) -> None:
        config = self.config_with("http", "0.0.0.0", 7001)
        # pose_udp listens on UDP 0.0.0.0:7001 in the same process.
        self.assertEqual([], config.listen_conflicts())
        self.assertIsNone(config.validate())

    def test_the_waypoint_target_is_not_a_listener(self) -> None:
        config = NetworkConfig()
        self.assertEqual(7010, config.waypoint.port)
        self.assertEqual("168.130.182.10", config.waypoint.host)
        self.assertFalse(config.waypoint.is_listen)
        self.assertEqual([], config.listen_conflicts())
        self.assertFalse(hasattr(config, "relay_tcp"))


class ClientDefaultTests(_CleanEnv):
    """The components take their endpoints from the same document."""

    def test_the_cimg_receiver_defaults_to_the_configured_endpoint(self) -> None:
        """``CimgConfig`` carried its own ``0.0.0.0:30106``.  The pipeline always
        passes the network values, so the default only reaches tests and tools -
        which makes a drifting default worse, not better: they would bind a port
        the receiver is not using."""
        from camera.cimg_source import CimgConfig

        endpoint = load_network_config().cimg
        config = CimgConfig()
        self.assertEqual((endpoint.host, endpoint.port), (config.host, config.port))
        self.assertEqual(("0.0.0.0", 30106), (config.host, config.port))

    def test_the_pipeline_builds_the_receiver_from_the_network_section(self) -> None:
        from config.loader import AppConfig

        os.environ["ALPAMAYO_CIMG_HOST"] = "127.0.0.1"
        os.environ["ALPAMAYO_CIMG_PORT"] = "39106"
        settings = AppConfig.from_env()
        self.assertEqual(39106, settings.network.cimg.port)
        # Imported here, inside the stub: the module-level block evicted
        # ``pipeline`` when it closed, so naming it after the fact re-executes
        # it against the real cv2 (tests/_cv2_isolation.py, "Usage rule").
        with cv2_stub():
            import pipeline

            built = pipeline._cimg_config(settings)
        self.assertEqual(("127.0.0.1", 39106), (built.host, built.port))
        self.assertEqual((1, 2, 0, 4), tuple(built.camera_indices))

    def test_the_offline_test_client_defaults_to_the_configured_url(self) -> None:
        from tools.request_model_test import parse_args

        self.assertEqual(
            load_network_config().package_test_url, parse_args(["sample.json"]).url
        )
        self.assertEqual(
            "http://127.0.0.1:8091/infer", parse_args(["sample.json"]).url
        )

    def test_the_offline_test_client_follows_the_document_and_the_environment(self) -> None:
        from tools import request_model_test

        path = self.write_document(
            {"http_client": {"package_test_url": "http://10.0.0.7:9000/infer"}}
        )
        os.environ["ALPAMAYO_CONFIG"] = path
        self.assertEqual(
            "http://10.0.0.7:9000/infer", request_model_test.parse_args(["s.json"]).url
        )
        os.environ["ALPAMAYO_PACKAGE_TEST_URL"] = "http://10.0.0.8:9000/infer"
        self.assertEqual(
            "http://10.0.0.8:9000/infer", request_model_test.parse_args(["s.json"]).url
        )
        # An explicit flag still wins over both.
        self.assertEqual(
            "http://10.0.0.9:9000/infer",
            request_model_test.parse_args(
                ["s.json", "--url", "http://10.0.0.9:9000/infer"]
            ).url,
        )


class StartupErrorTests(_CleanEnv):
    def test_a_bad_endpoint_fails_while_loading_the_app_config(self) -> None:
        """The failure has to happen before a socket exists, not at bind time."""
        from config.loader import AppConfig

        os.environ["ALPAMAYO_CIMG_PORT"] = "0"
        with self.assertRaises(ValueError) as caught:
            AppConfig.from_env()
        self.assertIn("network.cimg.port", str(caught.exception))

    def test_the_flat_app_fields_follow_the_network_section(self) -> None:
        from config.loader import AppConfig

        os.environ["ALPAMAYO_CIMG_PORT"] = "30109"
        os.environ["ALPAMAYO_POSE_UDP_HOST"] = "127.0.0.1"
        settings = AppConfig.from_env()
        self.assertEqual(30109, settings.cimg_port)
        self.assertEqual(30109, settings.network.cimg.port)
        self.assertEqual("127.0.0.1", settings.pose_udp_host)

class RetiredRelayTests(_CleanEnv):
    """Old PC-relay settings fail at load time instead of being ignored."""

    def test_a_relay_section_is_rejected(self) -> None:
        path = self.write_document(
            {"relay": {"tcp": {"host": "0.0.0.0", "port": 17100}}}
        )
        with self.assertRaises(ValueError) as caught:
            load_network_config(path)
        self.assertIn("network.relay", str(caught.exception))

    def test_a_retired_environment_variable_is_rejected(self) -> None:
        os.environ["RELAY_HOST"] = "127.0.0.1"
        with self.assertRaises(ValueError) as caught:
            load_network_config()
        self.assertIn("RELAY_HOST", str(caught.exception))

    def test_the_old_downlink_waypoint_is_rejected(self) -> None:
        path = self.write_document(
            {"waypoint": {"host": "127.0.0.1", "port": 7003}}
        )
        with self.assertRaises(ValueError) as caught:
            load_network_config(path)
        self.assertIn("127.0.0.1:7003", str(caught.exception))

    def test_loopback_is_allowed_for_a_non_relay_waypoint_port(self) -> None:
        config = load_network_config(
            self.write_document({"waypoint": {"host": "127.0.0.1", "port": 7010}})
        )
        self.assertEqual(("127.0.0.1", 7010), (config.waypoint.host, config.waypoint.port))


if __name__ == "__main__":
    unittest.main()
