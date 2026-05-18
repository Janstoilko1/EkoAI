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
    Linear(128→3)                      ← izhodni sloj (3 razredi)"""

from typing import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from matplotlib import pyplot as plt
import math
from pathlib import Path

RAZREDI = {"steklo" : 0 , "embalaza" : 1, "papir" : 2}

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.linear1 = nn.Linear(64*4*4, 128)
        self.linear2 = nn.Linear(128, 3)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)

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
        self.eval()  # ← izklopi dropout
        with torch.no_grad():  # ← ne računaj gradientov
            y_hat = self(x)
            probabilities = torch.softmax(y_hat, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class, probabilities
if __name__ == "__main__":
    obdelana_mapa = "odpadki_obdelani"

    X=[]
    Y=[]

    for razred in RAZREDI:
        mapa = Path(obdelana_mapa)/razred

    
                
        for pot in mapa.glob("*.npy"):
    
            spec = np.load(pot)
            spec = torch.tensor(np.array(spec), dtype=torch.float) / 255.0
            X.append(spec.unsqueeze(0).unsqueeze(0))
            Y.append(RAZREDI[razred])


    X = torch.cat(X,dim = 0)
    Y = torch.tensor(Y, dtype=torch.long)

    

    learningRate = 0.001
    epochs = 10000
    errorThreshold = 0.001
    
    model = NeuralNetwork()

    optimizer = optim.Adam(model.parameters(), lr = learningRate)

    losses = []

    start = time.time()

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        y_hat = model(X)

        loss = loss_fn(y_hat, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
            
        if loss.item() < errorThreshold:
            print("Učenje zaključeno, napaka je dovolj majhna.")
            break

        
    end = time.time()

    
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Iteracija")
    plt.ylabel("Vrednost izgube")
    plt.title(f"Potek učenja - čas: {end - start:.2f} s")
    plt.show()

    RAZREDI_INV = {v: k for k, v in RAZREDI.items()}
    # {0: "steklo", 1: "embalaza", 2: "papir"}

    # prediction na enem posnetku
    spec = np.load("odpadki_obdelani\\embalaza\\konzerva1_freq.npy")
    spec = torch.tensor(spec, dtype=torch.float32) / 255.0
    spec = spec.unsqueeze(0).unsqueeze(0)  # (1, 1, 129, 26)

    predicted_class, probabilities = model.predict(spec)

    print(f"Napoved: {RAZREDI_INV[predicted_class.item()]}")
    print(f"Verjetnosti: steklo={probabilities[0,0]:.2f}, embalaza={probabilities[0,1]:.2f}, papir={probabilities[0,2]:.2f}")