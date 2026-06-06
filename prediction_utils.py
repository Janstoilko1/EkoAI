from pathlib import Path

import numpy as np
import torch

import signal_processing as sp
from neural_network import NeuralNetwork, PODRAZREDI, PODRAZRED_V_RAZRED


MODEL_PATH = "model9.pth"
PODRAZREDI_INV = {v: k for k, v in PODRAZREDI.items()}


def parse_live_signal(data: bytes) -> tuple[np.ndarray, float]:
    """
    Iz binarnih podatkov STM32 loga naredi dekodiran audio signal in vzorčno frekvenco.
    """

    if data is None:
        raise ValueError("Ni podatkov za obdelavo.")

    if len(data) < 20:
        raise ValueError("Datoteka je premajhna.")

    if b"ERROR" in data[:200]:
        raise ValueError("Datoteka vsebuje ERROR tekst.")

    # Pozor: to uporablja globalne sezname v signal_processing.py
    sp.id = []
    sp.chunks = []
    sp.timestamps = []

    sp.separate(data)

    if len(sp.chunks) == 0:
        raise ValueError("V datoteki ni uporabnih chunkov.")

    id_arr = np.array(sp.id)
    ts_arr = np.array(sp.timestamps)
    ch_arr = np.array(sp.chunks, dtype=object)

    packets = []

    for pid, pts, pchunk in zip(id_arr, ts_arr, ch_arr):
        pdata = np.array(pchunk, dtype=np.int16).flatten()
        packets.append(sp.Paket(pid, pts, pdata))

    signal, Fvz = sp.sestavi_podatke(packets)

    remove = round(Fvz * 0.18)
    signal = signal[remove:]

    abs_idx = np.clip(
        np.abs(signal).astype(np.int32),
        0,
        len(sp.ALAW_DECODE_TABLE) - 1
    )

    decoded = sp.ALAW_DECODE_TABLE[abs_idx].astype(np.int16)
    decoded[signal < 0] = -decoded[signal < 0]

    return decoded, float(Fvz)


def izracunaj_spektrogram(signal: np.ndarray, Fvz: float):
    """
    Iz audio signala naredi spektrogram.
    """

    start, end = sp.najdi_dogodek(signal, Fvz)

    if end - start != sp.EVENT_SIZE:
        if len(signal) >= sp.EVENT_SIZE:
            start, end = 0, sp.EVENT_SIZE
        else:
            raise ValueError(
                f"Signal je prekratek ({len(signal)} vzorcev, potrebno {sp.EVENT_SIZE})."
            )

    spec = sp.signal_v_spektogram(
        signal,
        start,
        end,
        Fvz,
        normalize_16bit=True
    )

    return spec, start, end


def klasificiraj(spec: np.ndarray, model: NeuralNetwork, device: torch.device):
    """
    Požene spektrogram skozi nevronsko mrežo.
    """

    tensor = torch.tensor(spec, dtype=torch.float32) / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_class, probabilities = model.predict(tensor)

    cls = predicted_class.item()
    conf = probabilities[0, cls].item()

    return cls, conf


class WastePredictor:
    """
    Objekt, ki enkrat naloži model in potem dela predikcije nad fileData.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = NeuralNetwork().to(self.device)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict_from_bytes(self, file_data: bytes) -> dict:
        """
        Glavna funkcija za service.py.
        Vhod: fileData iz STM32.
        Izhod: dict z razredom, podrazredom, confidence in spektrogramom.
        """

        signal, Fvz = parse_live_signal(file_data)

        spec, start, end = izracunaj_spektrogram(signal, Fvz)

        cls, conf = klasificiraj(spec, self.model, self.device)

        if cls in PODRAZREDI_INV:
            podrazred = PODRAZREDI_INV[cls]
            razred = PODRAZRED_V_RAZRED.get(podrazred, "neznano")
        else:
            podrazred = f"neznan razred ({cls})"
            razred = "neznano"

        return {
            "razred": razred,
            "podrazred": podrazred,
            "confidence": conf,
            "class_id": cls,
            "Fvz": Fvz,
            "start": start,
            "end": end,
            "spectrogram": spec,
            "signal": signal,
        }