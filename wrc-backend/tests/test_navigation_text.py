from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config.loader import load_navigation_text


class NavigationTextConfigTests(unittest.TestCase):
    def _write_config(self, document: dict) -> str:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "backend.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return str(path)

    def test_missing_field_defaults_to_empty_string(self) -> None:
        path = self._write_config({"time_sync": {"enabled": True}})
        with patch.dict(os.environ, {"ALPAMAYO_CONFIG": path}, clear=True):
            self.assertEqual(load_navigation_text(), "")

    def test_configured_navigation_text_is_loaded(self) -> None:
        path = self._write_config({"navigation_text": " Turn left after the gate. "})
        with patch.dict(os.environ, {"ALPAMAYO_CONFIG": path}, clear=True):
            self.assertEqual(load_navigation_text(), "Turn left after the gate.")

    def test_settings_defaults_to_empty_when_config_value_is_empty(self) -> None:
        path = self._write_config({"navigation_text": ""})
        env = {
            "ALPAMAYO_CONFIG": path,
            "ALPAMAYO_MOCK_MODE": "1",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(load_navigation_text(), "")

    def test_environment_override_remains_compatible(self) -> None:
        path = self._write_config({"navigation_text": "from config"})
        env = {
            "ALPAMAYO_CONFIG": path,
            "ALPAMAYO_NAV_TEXT": "from environment",
            "ALPAMAYO_MOCK_MODE": "1",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(load_navigation_text(), "from environment")

if __name__ == "__main__":
    unittest.main()
