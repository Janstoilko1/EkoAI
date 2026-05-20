import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MAPA = Path("odpadki_obdelani")

SUFFIXES = [
    ("",                 "original"),
    ("_freq",            "original + freq mask"),
    ("_sum",             "šum"),
    ("_sum_freq",        "šum + freq mask"),
    ("_odmev",           "odmev"),
    ("_odmev_freq",      "odmev + freq mask"),
    ("_inv",             "inverzija"),
    ("_inv_freq",        "inverzija + freq mask"),
    ("_sum_odmev",       "šum + odmev"),
    ("_sum_odmev_freq",  "šum + odmev + freq mask"),
    ("_sum_inv",         "šum + inverzija"),
    ("_sum_inv_freq",    "šum + inverzija + freq mask"),
    ("_odmev_inv",       "odmev + inverzija"),
    ("_odmev_inv_freq",  "odmev + inverzija + freq mask"),
    ("_sum_odmev_inv",       "šum + odmev + inverzija"),
    ("_sum_odmev_inv_freq",  "šum + odmev + inverzija + freq mask"),
]

def najdi_vzorce():
    vzorci = []
    for pot in sorted(MAPA.rglob("*.npy")):
        ime = pot.stem
        if not any(ime.endswith(s) for s, _ in SUFFIXES if s):
            vzorci.append(pot)
    return vzorci

def prikazi(osnovna_pot: Path):
    ime = osnovna_pot.stem
    mapa = osnovna_pot.parent

    slike = []
    naslovi = []
    for suffix, opis in SUFFIXES:
        pot = mapa / (ime + suffix + ".npy")
        if pot.exists():
            slike.append(np.load(pot))
            naslovi.append(opis)

    n = len(slike)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for i, (slika, naslov) in enumerate(zip(slike, naslovi)):
        axes[i].imshow(slika, aspect="auto", origin="lower", cmap="inferno")
        axes[i].set_title(naslov, fontsize=9)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    razred = osnovna_pot.parent.name
    fig.suptitle(f"{razred} / {ime}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    vzorci = najdi_vzorce()
    if not vzorci:
        print("Ni najdenih vzorcev v", MAPA)
    else:
        print("Najdeni vzorci:")
        for i, v in enumerate(vzorci):
            print(f"  [{i}] {v.parent.name}/{v.stem}")

        izbira = input("\nVnesi številko vzorca (Enter za prvega): ").strip()
        idx = int(izbira) if izbira else 0
        prikazi(vzorci[idx])