"""Tests for the prediction path that service.py relies on.

This covers everything service.py does *except* the STM32 serial link: loading
the model and running real `.BIN` recordings through the full decode →
spectrogram → CNN pipeline (the same `WastePredictor.predict_from_bytes` call
that service.make_prediction wraps). No hardware, no GUI, no manual input.

service.py itself is intentionally NOT imported: importing it loads the model
at module scope and runs `mkdir` on a hardcoded absolute path, which is a side
effect we don't want in a test. We test the underlying logic directly instead.
"""

import glob
import os
import struct

os.environ.setdefault("MPLBACKEND", "Agg")  # never open a window

import numpy as np
import pytest

from neural_network import PODRAZRED_V_RAZRED, PODRAZREDI, RAZREDI
from prediction_utils import WastePredictor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Errors that a malformed/corrupt recording may legitimately raise while parsing.
PARSE_ERRORS = (ValueError, struct.error)

EXPECTED_KEYS = {
    "razred", "podrazred", "confidence", "class_id",
    "Fvz", "start", "end", "spectrogram", "signal",
}


def _resolve_model_path() -> str | None:
    for candidate in ("model9.pth", os.path.join("MODELS", "model9.pth")):
        path = os.path.join(REPO_ROOT, candidate)
        if os.path.exists(path):
            return path
    return None


@pytest.fixture(scope="session")
def predictor() -> WastePredictor:
    model_path = _resolve_model_path()
    if model_path is None:
        pytest.skip("model9.pth not found (checked repo root and MODELS/).")
    return WastePredictor(model_path)


@pytest.fixture(scope="session")
def bin_files() -> list[str]:
    files = sorted(glob.glob(os.path.join(REPO_ROOT, "Audio_logs", "**", "*.BIN"),
                             recursive=True))
    if not files:
        pytest.skip("No .BIN sample files available (Audio_logs/ empty).")
    return files


@pytest.fixture(scope="session")
def good_result(predictor, bin_files) -> dict:
    """Prediction for the first recording that parses cleanly."""
    for path in bin_files:
        with open(path, "rb") as f:
            data = f.read()
        try:
            return predictor.predict_from_bytes(data)
        except PARSE_ERRORS:
            continue
    pytest.skip("No .BIN file could be parsed.")


# --- structure / types of a single prediction ---


def test_result_has_all_expected_keys(good_result):
    assert EXPECTED_KEYS <= set(good_result)


def test_confidence_is_a_probability(good_result):
    assert 0.0 <= good_result["confidence"] <= 1.0


def test_class_id_is_an_int(good_result):
    assert isinstance(good_result["class_id"], int)


def test_signal_is_int16(good_result):
    assert isinstance(good_result["signal"], np.ndarray)
    assert good_result["signal"].dtype == np.int16
    assert good_result["signal"].size > 0


def test_spectrogram_is_2d_normalised(good_result):
    spec = good_result["spectrogram"]
    assert spec.ndim == 2
    assert spec.dtype == np.float32
    assert spec.min() >= 0.0
    assert spec.max() <= 255.0


def test_labels_are_consistent(good_result):
    """A non-'neznano' result must name a real sub-class whose parent class
    matches `razred` via the PODRAZRED_V_RAZRED mapping."""
    razred = good_result["razred"]
    podrazred = good_result["podrazred"]
    if razred != "neznano":
        assert podrazred in PODRAZREDI
        assert razred in RAZREDI
        assert PODRAZRED_V_RAZRED[podrazred] == razred


# --- behaviour ---


def test_prediction_is_deterministic(predictor, good_result, bin_files):
    """Same bytes → same prediction (model is in eval mode, no dropout noise)."""
    # Re-read the same file that produced good_result by matching on signal.
    for path in bin_files:
        with open(path, "rb") as f:
            data = f.read()
        try:
            first = predictor.predict_from_bytes(data)
        except PARSE_ERRORS:
            continue
        second = predictor.predict_from_bytes(data)
        assert first["class_id"] == second["class_id"]
        assert first["confidence"] == pytest.approx(second["confidence"])
        return
    pytest.skip("No parseable file for determinism check.")


def test_most_recordings_predict_without_unexpected_errors(predictor, bin_files):
    """Sweep the whole dataset: the vast majority must classify, and any
    failures must be parse errors (corrupt files), never model/logic crashes."""
    ok = 0
    for path in bin_files:
        with open(path, "rb") as f:
            data = f.read()
        try:
            result = predictor.predict_from_bytes(data)
        except PARSE_ERRORS:
            continue  # tolerated: known-corrupt recordings
        assert EXPECTED_KEYS <= set(result)
        ok += 1

    assert ok >= 0.9 * len(bin_files), (
        f"only {ok}/{len(bin_files)} recordings classified"
    )
