"""Smoke + unit tests for the audio pipeline's pure functions.

These run headless (no GUI, no STM32 hardware) against a real `.BIN` sample
shipped in `Audio_logs/`, so they exercise the actual parsing/decoding path
end to end without any device. GUI/serial modules (service.py, gui.py) are
intentionally not imported here.
"""

import glob
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # never open a window in CI

import numpy as np
import pytest

import signal_processing as sp
from prediction_utils import izracunaj_spektrogram, parse_live_signal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def sample_bin_bytes() -> bytes:
    """Bytes of the first real recording found under Audio_logs/."""
    pattern = os.path.join(REPO_ROOT, "Audio_logs", "**", "*.BIN")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        pytest.skip("No .BIN sample files available (Audio_logs/ empty).")
    with open(files[0], "rb") as f:
        return f.read()


# --- protocol primitives (no hardware, no data needed) ---


def test_crc_of_empty_is_init():
    assert sp.compute_crc(b"") == 0xFFFF


def test_crc_is_deterministic():
    payload = bytes(range(32))
    assert sp.compute_crc(payload) == sp.compute_crc(payload)


def test_unstuff_decodes_escape_pair():
    # 0xFE marker means: next byte is XOR'd with 0xFE.
    assert sp.unstuff(b"\xFE\x01", 2) == bytes([0xFE ^ 0x01])


def test_unstuff_passthrough_without_marker():
    assert sp.unstuff(b"\x01\x02\x03", 3) == b"\x01\x02\x03"


def test_alaw_table_is_monotonic_and_int16():
    table = sp.ALAW_DECODE_TABLE
    assert table.dtype == np.int16
    assert np.all(np.diff(table) > 0)  # strictly increasing magnitudes


# --- full parse against a real recording ---


def test_parse_live_signal_returns_int16_and_positive_rate(sample_bin_bytes):
    signal, fvz = parse_live_signal(sample_bin_bytes)
    assert signal.dtype == np.int16
    assert signal.size > 0
    assert fvz > 0


def test_parse_is_idempotent_despite_global_parser_state(sample_bin_bytes):
    """Guards the global-state reset gotcha in signal_processing.

    `separate`/`process` accumulate into module-level lists; if a caller
    forgets to reset them, a second parse would grow. Equal results prove
    the reset in parse_live_signal works.
    """
    sig1, fvz1 = parse_live_signal(sample_bin_bytes)
    sig2, fvz2 = parse_live_signal(sample_bin_bytes)
    assert sig1.size == sig2.size
    assert fvz1 == fvz2
    assert np.array_equal(sig1, sig2)


# --- event detection + spectrogram ---


def test_event_window_is_fixed_size(sample_bin_bytes):
    signal, fvz = parse_live_signal(sample_bin_bytes)
    _, start, end = izracunaj_spektrogram(signal, fvz)
    assert end - start == sp.EVENT_SIZE


def test_spectrogram_is_normalised_0_255(sample_bin_bytes):
    signal, fvz = parse_live_signal(sample_bin_bytes)
    spec, _, _ = izracunaj_spektrogram(signal, fvz)
    assert spec.dtype == np.float32
    assert spec.ndim == 2
    assert spec.min() >= 0.0
    assert spec.max() <= 255.0


def test_najdi_dogodek_clamps_to_signal_for_short_input():
    # A short synthetic signal must not return an out-of-range window.
    short = np.zeros(sp.EVENT_SIZE // 2, dtype=np.int16)
    start, end = sp.najdi_dogodek(short, Fvz=8000.0)
    assert 0 <= start <= end <= len(short)
