import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml


@pytest.fixture(autouse=True)
def _mock_hardware_deps():
    """Temporarily mock hardware deps only while importing main."""
    mocks = {}
    for mod in ["cv2", "mido", "ultralytics"]:
        if mod not in sys.modules:
            mocks[mod] = MagicMock()
            sys.modules[mod] = mocks[mod]
    yield
    for mod in mocks:
        del sys.modules[mod]


def _import_main():
    """Import main module with hardware deps mocked."""
    import importlib
    if "main" in sys.modules:
        return importlib.reload(sys.modules["main"])
    import main
    return main


class TestLoadConfig:
    def test_loads_valid_config(self):
        mod = _import_main()
        data = {"vision": {"model": "yolov8n-pose"}, "silence": {"timeout_ms": 500}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            f.flush()
            result = mod.load_config(f.name)
        os.unlink(f.name)
        assert result["vision"]["model"] == "yolov8n-pose"
        assert result["silence"]["timeout_ms"] == 500

    def test_missing_file_exits(self):
        mod = _import_main()
        with pytest.raises(SystemExit):
            mod.load_config("/nonexistent/path/config.yaml")

    def test_invalid_yaml_exits(self):
        mod = _import_main()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("invalid: yaml: [unterminated\n")
            f.flush()
            with pytest.raises(SystemExit):
                mod.load_config(f.name)
        os.unlink(f.name)


class TestParseArgs:
    def test_defaults(self, monkeypatch):
        mod = _import_main()
        monkeypatch.setattr("sys.argv", ["main.py"])
        args = mod.parse_args()
        assert args.mode == "midi"
        assert args.midi_mode == "gesture"
        assert args.config == "config.yaml"

    def test_custom_flags(self, monkeypatch):
        mod = _import_main()
        monkeypatch.setattr(
            "sys.argv",
            ["main.py", "--mode", "midi", "--midi-mode", "musical", "--config", "custom.yaml"],
        )
        args = mod.parse_args()
        assert args.config == "custom.yaml"

    def test_invalid_mode_exits(self, monkeypatch):
        mod = _import_main()
        monkeypatch.setattr("sys.argv", ["main.py", "--mode", "invalid"])
        with pytest.raises(SystemExit):
            mod.parse_args()
