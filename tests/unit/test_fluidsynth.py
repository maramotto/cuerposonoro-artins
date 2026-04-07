import pytest
from unittest.mock import MagicMock, call, patch
from audio.fluidsynth import FluidsynthManager


class TestFluidsynthManager:
    @patch("audio.fluidsynth.subprocess")
    def test_start_launches_process(self, mock_subprocess):
        mock_subprocess.Popen.return_value = MagicMock()
        mgr = FluidsynthManager(
            soundfont="JJazzLab-SoundFont.sf2",
            gain=0.8,
            sample_rate=44100,
        )
        mgr.start()
        mock_subprocess.Popen.assert_called_once()
        args = mock_subprocess.Popen.call_args[0][0]
        assert "fluidsynth" in args[0]
        assert "JJazzLab-SoundFont.sf2" in args

    @patch("audio.fluidsynth.subprocess")
    def test_stop_terminates_process(self, mock_subprocess):
        mock_proc = MagicMock()
        mock_subprocess.Popen.return_value = mock_proc
        mgr = FluidsynthManager(
            soundfont="JJazzLab-SoundFont.sf2",
            gain=0.8,
            sample_rate=44100,
        )
        mgr.start()
        mgr.stop()
        mock_proc.terminate.assert_called_once()

    @patch("audio.fluidsynth.subprocess")
    def test_stop_without_start_is_safe(self, mock_subprocess):
        mgr = FluidsynthManager(
            soundfont="JJazzLab-SoundFont.sf2",
            gain=0.8,
            sample_rate=44100,
        )
        mgr.stop()  # Should not raise

    @patch("audio.fluidsynth.subprocess")
    def test_stop_kills_on_timeout(self, mock_subprocess):
        """If Fluidsynth hangs on terminate, kill it and still clear state."""
        import subprocess as real_subprocess
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = [
            real_subprocess.TimeoutExpired("fluidsynth", 5),  # first wait
            None,  # wait after kill
        ]
        mock_subprocess.Popen.return_value = mock_proc
        mock_subprocess.TimeoutExpired = real_subprocess.TimeoutExpired
        mgr = FluidsynthManager(
            soundfont="JJazzLab-SoundFont.sf2",
            gain=0.8,
            sample_rate=44100,
        )
        mgr.start()
        mgr.stop()
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert not mgr.running

    @patch("audio.fluidsynth.subprocess")
    def test_gain_in_args(self, mock_subprocess):
        mock_subprocess.Popen.return_value = MagicMock()
        mgr = FluidsynthManager(
            soundfont="JJazzLab-SoundFont.sf2",
            gain=0.5,
            sample_rate=44100,
        )
        mgr.start()
        args = mock_subprocess.Popen.call_args[0][0]
        gain_idx = args.index("-g")
        assert args[gain_idx + 1] == "0.5"


# --- aconnect / connect_midi_port tests ---

ACONNECT_OUTPUT = """\
client 0: 'System' [type=kernel]
    0 'Timer           '
    1 'Announce        '
client 14: 'Midi Through' [type=kernel]
    0 'Midi Through Port-0'
client 128: 'FLUID Synth (31337)' [type=user,pid=31337]
    0 'Synth input port (31337:0)'
client 129: 'cuerposonoro' [type=user,pid=9999]
    0 'cuerposonoro    '
"""

ACONNECT_OUTPUT_NO_FLUID = """\
client 0: 'System' [type=kernel]
    0 'Timer           '
client 14: 'Midi Through' [type=kernel]
    0 'Midi Through Port-0'
client 129: 'cuerposonoro' [type=user,pid=9999]
    0 'cuerposonoro    '
"""


class TestConnectMidiPort:
    def _make_manager(self) -> FluidsynthManager:
        return FluidsynthManager(
            soundfont="JJazzLab-SoundFont.sf2",
            gain=0.8,
            sample_rate=44100,
        )

    @patch("audio.fluidsynth.time.sleep")
    @patch("audio.fluidsynth.subprocess")
    def test_connect_calls_aconnect_with_correct_ports(
        self, mock_subprocess, mock_sleep
    ):
        """connect_midi_port must call aconnect <virtual_port> <fluid_port>."""
        mock_subprocess.Popen.return_value = MagicMock()
        # aconnect -l returns the listing; aconnect connect call succeeds
        mock_subprocess.run.side_effect = [
            MagicMock(stdout=ACONNECT_OUTPUT, returncode=0),  # aconnect -l
            MagicMock(returncode=0),  # aconnect connect
        ]
        mgr = self._make_manager()
        mgr.start()
        mgr.connect_midi_port("cuerposonoro")

        # Second call to subprocess.run should be the aconnect connect
        connect_call = mock_subprocess.run.call_args_list[1]
        args = connect_call[0][0]
        assert args[0] == "aconnect"
        assert "129:0" in args  # source: cuerposonoro
        assert "128:0" in args  # destination: FLUID Synth
        # source comes before destination in aconnect
        src_idx = args.index("129:0")
        dst_idx = args.index("128:0")
        assert src_idx < dst_idx

    @patch("audio.fluidsynth.time.sleep")
    @patch("audio.fluidsynth.subprocess")
    def test_connect_retries_when_fluid_port_not_found(
        self, mock_subprocess, mock_sleep
    ):
        """connect_midi_port retries when Fluidsynth port is not yet visible."""
        mock_subprocess.Popen.return_value = MagicMock()
        mock_subprocess.run.side_effect = [
            MagicMock(stdout=ACONNECT_OUTPUT_NO_FLUID, returncode=0),  # attempt 1
            MagicMock(stdout=ACONNECT_OUTPUT, returncode=0),  # attempt 2
            MagicMock(returncode=0),  # aconnect connect
        ]
        mgr = self._make_manager()
        mgr.start()
        mgr.connect_midi_port("cuerposonoro")

        # Should have called aconnect -l twice, then aconnect connect once
        assert mock_subprocess.run.call_count == 3
        mock_sleep.assert_called()

    @patch("audio.fluidsynth.time.sleep")
    @patch("audio.fluidsynth.subprocess")
    def test_connect_raises_after_max_retries(
        self, mock_subprocess, mock_sleep
    ):
        """connect_midi_port raises RuntimeError after exhausting retries."""
        mock_subprocess.Popen.return_value = MagicMock()
        # Always return no FLUID Synth port
        mock_subprocess.run.return_value = MagicMock(
            stdout=ACONNECT_OUTPUT_NO_FLUID, returncode=0
        )
        mgr = self._make_manager()
        mgr.start()

        with pytest.raises(RuntimeError, match="FLUID Synth"):
            mgr.connect_midi_port("cuerposonoro")

    @patch("audio.fluidsynth.time.sleep")
    @patch("audio.fluidsynth.subprocess")
    def test_connect_raises_when_virtual_port_not_found(
        self, mock_subprocess, mock_sleep
    ):
        """connect_midi_port raises RuntimeError if the virtual port is missing."""
        mock_subprocess.Popen.return_value = MagicMock()
        # Fluidsynth port exists but virtual port does not
        aconnect_no_virtual = """\
client 0: 'System' [type=kernel]
    0 'Timer           '
client 128: 'FLUID Synth (31337)' [type=user,pid=31337]
    0 'Synth input port (31337:0)'
"""
        mock_subprocess.run.return_value = MagicMock(
            stdout=aconnect_no_virtual, returncode=0
        )
        mgr = self._make_manager()
        mgr.start()

        with pytest.raises(RuntimeError, match="cuerposonoro"):
            mgr.connect_midi_port("cuerposonoro")

    @patch("audio.fluidsynth.time.sleep")
    @patch("audio.fluidsynth.subprocess")
    def test_connect_raises_if_aconnect_fails(
        self, mock_subprocess, mock_sleep
    ):
        """connect_midi_port raises RuntimeError if aconnect returns non-zero."""
        mock_subprocess.Popen.return_value = MagicMock()
        mock_subprocess.run.side_effect = [
            MagicMock(stdout=ACONNECT_OUTPUT, returncode=0),  # aconnect -l
            MagicMock(returncode=1, stderr="Connection refused"),  # aconnect fails
        ]
        mgr = self._make_manager()
        mgr.start()

        with pytest.raises(RuntimeError, match="aconnect"):
            mgr.connect_midi_port("cuerposonoro")
