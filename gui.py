import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.io import wavfile

import signal_processing as sp
from neural_network import NeuralNetwork, PODRAZREDI, PODRAZRED_V_RAZRED


MODEL_PATH = "model9.pth"
PODRAZREDI_INV = {v: k for k, v in PODRAZREDI.items()}


def parse_bin_file(filepath: str) -> tuple[np.ndarray, float]:
    with open(filepath, "rb") as f:
        data = f.read()

    if len(data) < 20:
        raise ValueError("Datoteka je premajhna.")
    if b"ERROR" in data[:200]:
        raise ValueError("Datoteka vsebuje ERROR tekst.")

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

    abs_idx = np.clip(np.abs(signal).astype(np.int32),
                      0, len(sp.ALAW_DECODE_TABLE) - 1)
    decoded = sp.ALAW_DECODE_TABLE[abs_idx].astype(np.int16)
    decoded[signal < 0] = -decoded[signal < 0]
    return decoded, float(Fvz)

def parse_live_singal(data: bytes) -> tuple[np.ndarray, float]:
    if len(data) < 20:
        raise ValueError("Datoteka je premajhna.")
    if b"ERROR" in data[:200]:
        raise ValueError("Datoteka vsebuje ERROR tekst.")

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

    abs_idx = np.clip(np.abs(signal).astype(np.int32),
                      0, len(sp.ALAW_DECODE_TABLE) - 1)
    decoded = sp.ALAW_DECODE_TABLE[abs_idx].astype(np.int16)
    decoded[signal < 0] = -decoded[signal < 0]
    return decoded, float(Fvz)



def parse_wav_file(filepath: str) -> tuple[np.ndarray, float]:
    Fvz, signal = wavfile.read(filepath)
    if signal.ndim > 1:
        signal = signal[:, 0]
    return signal.astype(np.int16), float(Fvz)


def izracunaj_spektrogram(signal: np.ndarray, Fvz: float):
    start, end = sp.najdi_dogodek(signal, Fvz)
    if end - start != sp.EVENT_SIZE:
        if len(signal) >= sp.EVENT_SIZE:
            start, end = 0, sp.EVENT_SIZE
        else:
            raise ValueError(
                f"Signal je prekratek ({len(signal)} vzorcev, potrebno {sp.EVENT_SIZE})."
            )
    spec = sp.signal_v_spektogram(signal, start, end, Fvz, normalize_16bit=True)
    return spec, start, end


def klasificiraj(spec: np.ndarray, model: NeuralNetwork, device: torch.device):
    tensor = torch.tensor(spec, dtype=torch.float32) / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
    predicted_class, probabilities = model.predict(tensor)
    cls = predicted_class.item()
    conf = probabilities[0, cls].item()
    return cls, conf


class Aplikacija:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EkoAI - klasifikacija odpadkov")
        self.root.geometry("960x780")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralNetwork().to(self.device)
        try:
            state = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state)
        except FileNotFoundError:
            messagebox.showerror("Napaka", f"Ne najdem datoteke {MODEL_PATH}")
            self.root.destroy()
            return
        except Exception as e:
            messagebox.showerror("Napaka pri nalaganju modela",
                                 f"{type(e).__name__}: {e}")
            self.root.destroy()
            return
        self.model.eval()

        self.file_path_var = tk.StringVar(value="Datoteka ni izbrana.")
        self.razred_var = tk.StringVar(value="Razred: -")
        self.podrazred_var = tk.StringVar(value="Podrazred: -")
        self.prepricanost_var = tk.StringVar(value="Prepricanost: -")
        self.naprava_var = tk.StringVar(
            value=f"Naprava: {self.device}  |  Model: {MODEL_PATH}"
        )

        self._zgradi_layout()

    def _zgradi_layout(self):
        zgornji = ttk.Frame(self.root, padding=10)
        zgornji.pack(fill="x")

        ttk.Button(zgornji,
                   text="Naloži datoteko (.BIN / .wav)",
                   command=self.nalozi_datoteko).pack(side="left")
        ttk.Label(zgornji, textvariable=self.file_path_var,
                  wraplength=700).pack(side="left", padx=10)

        rezultat_okvir = ttk.LabelFrame(self.root, text="Rezultat", padding=10)
        rezultat_okvir.pack(fill="x", padx=10, pady=5)

        ttk.Label(rezultat_okvir, textvariable=self.razred_var,
                  font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(rezultat_okvir, textvariable=self.podrazred_var,
                  font=("Segoe UI", 13)).pack(anchor="w")
        ttk.Label(rezultat_okvir, textvariable=self.prepricanost_var,
                  font=("Segoe UI", 13)).pack(anchor="w")

        self.fig = plt.Figure(figsize=(9, 4.5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True,
                                         padx=10, pady=10)
        self.ax.set_title("Spektrogram (naloži datoteko)")
        self.ax.set_xlabel("Casovni okvir")
        self.ax.set_ylabel("Frekvencni bin")
        self.canvas.draw()

        spodnji = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        spodnji.pack(fill="x")
        ttk.Label(spodnji, textvariable=self.naprava_var,
                  foreground="#666").pack(anchor="w")

    def nalozi_datoteko(self):
        path = filedialog.askopenfilename(
            title="Izberi .BIN ali .wav datoteko",
            filetypes=[
                ("Audio datoteke", "*.BIN *.bin *.wav"),
                (".BIN datoteke", "*.BIN *.bin"),
                (".wav datoteke", "*.wav"),
                ("Vse datoteke", "*.*"),
            ],
        )
        if not path:
            return
        self.file_path_var.set(path)
        self.obdelaj(path)

    def obdelaj(self, path: str, data=None):
        try:
            """sufiks = Path(path).suffix.lower()
            if sufiks == ".wav":
                signal, Fvz = parse_wav_file(path)
            else:
                signal, Fvz = parse_bin_file(path)"""
            signal, Fvz = parse_live_singal(data)

            spec, _, _ = izracunaj_spektrogram(signal, Fvz)
            cls, conf = klasificiraj(spec, self.model, self.device)

            if cls in PODRAZREDI_INV:
                podrazred = PODRAZREDI_INV[cls]
                razred = PODRAZRED_V_RAZRED.get(podrazred, "neznano")
            else:
                podrazred = f"neznan razred ({cls})"
                razred = "neznano"

            self.razred_var.set(f"Razred: {razred}")
            self.podrazred_var.set(f"Podrazred: {podrazred}")
            self.prepricanost_var.set(f"Prepricanost: {conf * 100:.2f}%")

            self._prikazi_spektrogram(spec, Path(path).name)
        except Exception as e:
            messagebox.showerror("Napaka pri obdelavi",
                                 f"{type(e).__name__}: {e}")
            self.razred_var.set("Razred: -")
            self.podrazred_var.set("Podrazred: -")
            self.prepricanost_var.set("Prepricanost: -")

    def _prikazi_spektrogram(self, spec: np.ndarray, naslov: str):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        im = self.ax.imshow(spec, aspect="auto", origin="lower", cmap="inferno")
        self.ax.set_title(f"Spektrogram - {naslov}")
        self.ax.set_xlabel("Casovni okvir")
        self.ax.set_ylabel("Frekvencni bin")
        self.fig.colorbar(im, ax=self.ax)
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    Aplikacija(root)
    root.mainloop()
