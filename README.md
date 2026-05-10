# EkoAI
Sortiranje smeti na podlagi zvočnih signalov s pomočjo umetne inteligence.

## `obdelaj_vse(surova_mapa, izhodna_mapa)`

Pretvori vse surove posnetke v spektrograme in jih shrani kot `.npy` datoteke.

### Vhodna struktura map
surova_mapa/
├── embalaza/
│   ├── konzerva/
│   ├── plastenka/
│   └── ...
├── papir/
│   └── karton/
└── steklo/
└── zacimba_steklenica/

### Postopek obdelave vsakega posnetka
1. Prebere surove binarne podatke iz datoteke
2. Razstavi pakete (`separate`, `unstuff`, `unpack`)
3. Sestavi signal in izračuna vzorčno frekvenco (`sestavi_podatke`)
4. Dekodira A-law kompresijo (`ALAW_DECODE_TABLE`)
5. Izreže relevantni dogodek iz signala (`najdi_dogodek`)
6. Pretvori signal v spektrogram z STFT (`signal_v_spektrogram`)
7. Shrani spektrogram kot `.npy` datoteko

### Izhodna struktura map
izhodna_mapa/
├── embalaza/
│   ├── konzerva_posnetek1.npy
│   └── plastenka_posnetek1.npy
├── papir/
└── steklo/

### Uporaba
```python
obdelaj_vse("odpadki_surovi", "odpadki_obdelani")
```