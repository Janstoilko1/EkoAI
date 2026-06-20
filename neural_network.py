"""Implementacija Konvolucijske nevronske mreže.

   VHODNA PLAST -> 129x26 = 3354 nevronov

    Input (1, 129, 26)
     ↓
    Conv2d(1→16)   + ReLU + MaxPool    ← skriti sloj 1
     ↓
    Conv2d(16→32)  + ReLU + MaxPool    ← skriti sloj 2
     ↓
    Conv2d(32→64)  + ReLU + AvgPool    ← skriti sloj 3
     ↓
    Flatten
     ↓
    Linear(1024→128) + ReLU + Dropout  ← skriti sloj 4
     ↓
    Linear(128→11)                     ← izhodni sloj (11 podrazredov)"""

from typing import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
from matplotlib import pyplot as plt
import math
from pathlib import Path

PODRAZREDI = {
    "konzerva":            0,
    "plastenka":           1,
    "plocevinka":          2,
    "puding":              3,
    "tetrapak":            4,
    "karton":              5,
    "papirni_kozarcek":    6,
    "zmeckan_papir":       7,
    "Steklen_kozarcek_v2": 8,
    "stekleni_kozarcek":   9,
    "zacimba_steklenica":  10,
}

PODRAZRED_V_RAZRED = {
    "konzerva":            "embalaza",
    "plastenka":           "embalaza",
    "plocevinka":          "embalaza",
    "puding":              "embalaza",
    "tetrapak":            "embalaza",
    "karton":              "papir",
    "papirni_kozarcek":    "papir",
    "zmeckan_papir":       "papir",
    "Steklen_kozarcek_v2": "steklo",
    "stekleni_kozarcek":   "steklo",
    "zacimba_steklenica":  "steklo",
}

RAZREDI = {"steklo": 0, "embalaza": 1, "papir": 2}

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.linear1 = nn.Linear(64*4*4, 128)

        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, len(PODRAZREDI))

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)

        #pooling
        self.maxPool = nn.MaxPool2d(2,2)
        self.avgPool = nn.AdaptiveAvgPool2d((4,4))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxPool(x)

        x = self.relu(self.conv2(x))
        x = self.maxPool(x)

        x = self.relu(self.conv3(x))
        x = self.avgPool(x)

        x = self.flatten(x)

        x = self.relu(self.linear1(x))
        x = self.dropout(x)

        x = self.linear2(x)

        return x
    
    def learnNN(self):
        pass

    def predict(self, x):
        device = next(self.parameters()).device
        x = x.to(device)
        self.eval()
        with torch.no_grad():
            y_hat = self(x)
            probabilities = torch.softmax(y_hat, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class, probabilities

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Naprava: {device}")

    obdelana_mapa = "odpadki_obdelani"
    test_mapa = Path("test_data")

    # zberemo imena vseh testnih datotek, da jih izključimo iz učenja
    test_datoteke = set()
    for pot in test_mapa.glob("**/*.npy"):
        test_datoteke.add(pot.name)

    X = []
    Y = []

    for podrazred, podrazred_id in PODRAZREDI.items():
        razred = PODRAZRED_V_RAZRED[podrazred]
        mapa = Path(obdelana_mapa) / razred / podrazred
        if not mapa.exists():
            continue
        for pot in mapa.glob("*.npy"):
            if pot.name in test_datoteke:
                continue
            spec = np.load(pot)
            spec = torch.tensor(np.array(spec), dtype=torch.float) / 255.0
            X.append(spec.unsqueeze(0).unsqueeze(0))
            Y.append(podrazred_id)

    X = torch.cat(X, dim=0)
    Y = torch.tensor(Y, dtype=torch.long)

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    learningRate = 0.0001
    epochs = 10000
    errorThreshold = 0.01
    model = NeuralNetwork().to(device)

    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100)

    losses = []
    start = time.time()
    loss_fn = nn.CrossEntropyLoss()
    zadnji_procent = -1

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, Y_batch in loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            y_hat = model(X_batch)
            loss = loss_fn(y_hat, Y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)
        losses.append(avg_loss)

        procent = int((epoch + 1) / epochs * 100)
        if procent != zadnji_procent:
            print(f"\rUčenje: {procent}%  (izguba: {avg_loss:.6f})", end="", flush=True)
            zadnji_procent = procent

        if avg_loss < errorThreshold:
            print(f"\rUčenje: 100%  (izguba: {avg_loss:.6f})")
            print("Učenje zaključeno, napaka je dovolj majhna.")
            break
    else:
        print()

    end = time.time()

    torch.save(model.state_dict(), "model9.pth")
    print("Model shranjen v model9.pth")

    plt.plot(range(len(losses)), losses)
    plt.xlabel("Iteracija")
    plt.ylabel("Vrednost izgube")
    plt.title(f"Potek učenja - čas: {end - start:.2f} s")
    plt.show()

    PODRAZREDI_INV = {v: k for k, v in PODRAZREDI.items()}

    test_mapa = Path("test_data")
    pravilno_razred = 0
    pravilno_podrazred = 0
    skupaj = 0

    razred_pravilno = {r: 0 for r in RAZREDI}
    razred_skupaj   = {r: 0 for r in RAZREDI}

    print(f"\n=== Testiranje ===")
    for podrazred in sorted(PODRAZREDI):
        razred = PODRAZRED_V_RAZRED[podrazred]
        podmapa = test_mapa / razred / podrazred
        if not podmapa.exists():
            continue

        for pot in sorted(podmapa.glob("*.BIN.npy")):
            spec = np.load(pot)
            spec = torch.tensor(spec, dtype=torch.float32) / 255.0
            spec = spec.unsqueeze(0).unsqueeze(0).to(device)

            predicted_class, _ = model.predict(spec)
            napovedan_podrazred = PODRAZREDI_INV[predicted_class.item()]
            napovedan_razred    = PODRAZRED_V_RAZRED[napovedan_podrazred]

            pravilno_podrazred += int(napovedan_podrazred == podrazred)
            pravilno_razred    += int(napovedan_razred == razred)
            razred_pravilno[razred] += int(napovedan_razred == razred)
            razred_skupaj[razred]   += 1
            skupaj += 1

            print(f"{razred}/{podrazred}/{pot.name:<40} -> {napovedan_podrazred} ({napovedan_razred})"
                  f"  {'OK' if napovedan_razred == razred else 'NAPAKA'}")

    if skupaj > 0:
        print(f"\nTočnost (razred):    {pravilno_razred}/{skupaj} ({100 * pravilno_razred / skupaj:.1f}%)")
        print(f"Točnost (podrazred): {pravilno_podrazred}/{skupaj} ({100 * pravilno_podrazred / skupaj:.1f}%)")
        for razred in sorted(RAZREDI):
            st = razred_skupaj[razred]
            if st > 0:
                print(f"  {razred}: {razred_pravilno[razred]}/{st} ({100 * razred_pravilno[razred] / st:.1f}%)")
    else:
        print("Ni testnih datotek v", test_mapa)