import struct
import random
import shutil
from attr import dataclass
import numpy as np
import matplotlib.pyplot as plt
import string
import sounddevice as sd
import scipy.signal as sp_signal
from pathlib import Path
@dataclass
class Paket:
	id: int      
	time: float    
	data: np.ndarray

ALAW_DECODE_TABLE = np.array([
      8,   24,   40,   56,   72,   88,  104,  120,
    136,  152,  168,  184,  200,  216,  232,  248,
    264,  280,  296,  312,  328,  344,  360,  376,
    392,  408,  424,  440,  456,  472,  488,  504,
    528,  560,  592,  624,  656,  688,  720,  752,
    784,  816,  848,  880,  912,  944,  976, 1008,
   1056, 1120, 1184, 1248, 1312, 1376, 1440, 1504,
   1568, 1632, 1696, 1760, 1824, 1888, 1952, 2016,
   2112, 2240, 2368, 2496, 2624, 2752, 2880, 3008,
   3136, 3264, 3392, 3520, 3648, 3776, 3904, 4032,
   4224, 4480, 4736, 4992, 5248, 5504, 5760, 6016,
   6272, 6528, 6784, 7040, 7296, 7552, 7808, 8064,
   8448, 8960, 9472, 9984,10496,11008,11520,12032,
  12544,13056,13568,14080,14592,15104,15616,16128,
  16896,17920,18944,19968,20992,22016,23040,24064,
  25088,26112,27136,28160,29184,30208,31232,32256,
], dtype=np.int16)

EVENT_SIZE = 6000

RAZREDI = {"steklo" : 0 , "embalaza" : 1, "papir" : 2}


id = []
chunks = []
timestamps = []

def unstuff(byteData, size):
    unstuffed = []
    i=0
    while i<size:
        if i<=size-2:
            if byteData[i]==0xFE:
                unstuffed.append(0xFE^byteData[i+1])
                i+=2
                continue
        unstuffed.append(byteData[i])
        i+=1
    return bytes(unstuffed)

def unpack(data, counter):
    packet_counter=struct.unpack('<B', data[0:1])[0]
    if counter%254 != packet_counter:
        print("packet skipped", counter%254, packet_counter)
    process(data[1:len(data)])

def separate(data):
    packets = []
    start=0
    stop=0
    counter=-1
    while(start<len(data) and data[start:start+2]!=b'\xFF\xFF'):
        start+=1
    stop=start+1
    counter=struct.unpack('<B', data[start+2:start+3])[0] - 1

    while(stop<len(data)):

        while(stop<len(data) and data[stop:stop+2]!=b'\xFF\xFF'):
            stop+=1

        #zaporedna stevilka packeta
        counter+=1
        packets.append(unpack(data[start+2:stop], counter))

        if start>=len(data):
            break
        start=stop

        while(start<len(data) and data[start:start+2]!=b'\xFF\xFF'):
            start+=1
        stop=start+1

def crc16_update(crc: int, data: int) -> int:
    crc ^= data
    for _ in range(8):
        if crc & 1:
            crc = (crc >> 1) ^ 0xA001
        else:
            crc >>= 1
    return crc

def compute_crc(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc = crc16_update(crc, byte)
    return crc

def process(data):
    global id, chunks, timestamps
    payload = unstuff(data, len(data))

    #timestamp - 4 bytes
    timestamp = struct.unpack('<I',payload[0:4])[0]

    #packet size in bytes - 2 bytes
    packet_size = struct.unpack('<H', payload[4:6])[0]+1

    #CRC - 2 bytes
    crc = struct.unpack('<H', payload[6+packet_size:6+packet_size+2])[0]
    computed_crc = compute_crc(payload[:6+packet_size])
    if crc!=computed_crc:
        print("CRC mismatch", crc, computed_crc)
        return

    chunks_data = payload[6:6+packet_size]
    pos = 0
    while pos < len(chunks_data):
        #chunk ID - 1 byte
        chunk_id = chunks_data[pos]
        id.append(chunk_id)
        #chunk size-1 - 2 bytes
        chunk_size_enc = struct.unpack('<H', chunks_data[pos+1:pos+3])[0]
        chunk_size = chunk_size_enc + 1

        #reserved for future use - 1 byte
        reserved = chunks_data[pos+3]

        #chunk data - chunk_size
        chunk_data = chunks_data[pos+4:pos+4+chunk_size]

        samples = []
        for i in range(0, chunk_size):
            try:
                value = struct.unpack('<b', chunk_data[i:i+1])[0]
            except struct.error:
                print(f"Error unpacking data at position {i}")
                break
            samples.append(value)
        timestamps.append(timestamp)
        chunks.append(samples)
        pos += 4 + chunk_size




def najdi_dogodek(signal: np.ndarray, Fvz: float,
                  window_s: float = 0.025,
                  peak_fraction: float = 0.05) -> tuple[int, int]:
    """
    Returns (startInd, endInd) spanning exactly the samples where RMS energy
    exceeds peak_fraction * peak_RMS.
    """
    win = max(1, int(window_s * Fvz))
    mag = signal.astype(np.float64)

    # high-pass filter to remove DC drift (brez tega se slabo doloci dogodek)
    b, hp = 0.995, np.zeros_like(mag)
    for i in range(1, len(mag)):
        hp[i] = b * (hp[i-1] + mag[i] - mag[i-1])
    mag = np.abs(hp)

    #compute RMS energy (root mean square )
    n_wins = len(mag) // win
    energy = np.array([np.sqrt(np.mean(mag[i*win:(i+1)*win]**2)) for i in range(n_wins)])
    threshold = peak_fraction * energy.max()
    active = energy > threshold

    if not np.any(active):
        return 0, len(signal)

    first_win = int(np.argmax(active))
    start = first_win * win
    if start + EVENT_SIZE >len(signal):
        end = len(signal)
        start = len(signal) - EVENT_SIZE
    else:
        end = (start + EVENT_SIZE)
    return start, end

def augmentiraj_podatke(signal: np.ndarray, Fvz: float):
    sig = signal.astype(np.float64)

    noise = np.random.normal(0, 0.01 * np.max(np.abs(sig)), size=sig.shape)
    echo_kernel = np.exp(-np.arange(0, int(len(sig) * 0.1)) / 100)

    s_sum   = sig + noise
    s_odmev = np.convolve(sig, echo_kernel, mode='full')[:len(sig)]
    s_inv   = -sig

    s_sum_odmev = np.convolve(s_sum, echo_kernel, mode='full')[:len(sig)]
    s_sum_inv   = -s_sum
    s_odmev_inv = -s_odmev
    s_sum_odmev_inv = -s_sum_odmev

    return [
        (s_sum,         "_sum"),
        (s_odmev,       "_odmev"),
        (s_inv,         "_inv"),
        (s_sum_odmev,   "_sum_odmev"),
        (s_sum_inv,     "_sum_inv"),
        (s_odmev_inv,   "_odmev_inv"),
        (s_sum_odmev_inv, "_sum_odmev_inv"),
    ]



def prikazi_signal(signal: np.ndarray, naslov: string, startInd: int, endInd: int, Fvz: float, normalize_16bit: bool = False):
    if startInd is None:
        startInd = 0
    if endInd is None:
        endInd = len(signal)

    interval = signal[startInd:endInd]
    time = np.arange(startInd, endInd) / Fvz

    podatki =augmentiraj_podatke(interval, Fvz)
    for interval in podatki:
        if normalize_16bit:
            sig_float = interval.astype(np.float64)
            peak = np.max(sig_float)
            low = np.min(sig_float)
            if peak > 0:
                interval = (sig_float - low)/(peak - low) * 65535 - 32768
            interval = interval.astype(np.int16)

        fig, (ax_sig, ax_spec) = plt.subplots(2, 1, figsize=(12, 8))

        ax_sig.plot(time, interval, label="Value")
        ax_sig.set_xlabel("time [s]")
        ax_sig.set_ylabel("Amplitude")
        ax_sig.set_title(naslov)
        ax_sig.legend()

        # spectrogram (STFT)
        spec_sig = np.array(interval, dtype=np.float64)
            
        nperseg = min(256, max(16, len(spec_sig) // 8))
        f_spec, t_spec, Sxx = sp_signal.spectrogram(spec_sig, fs=Fvz, nperseg=nperseg)
        im = ax_spec.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="inferno")
        fig.colorbar(im, ax=ax_spec)
        ax_spec.set_xlabel("Time [s]")
        ax_spec.set_ylabel("Frequency [Hz]")
        ax_spec.set_title("Spectrogram (STFT)")

        plt.tight_layout()
        plt.show()

        spectogram_input = 10 * np.log10(Sxx + 1e-10)
        shrani_spektrogram(spectogram_input)


def sestavi_podatke(packets: np.ndarray, id: int = 4):
    timestamps = [packet.time for packet in packets if packet.id==id]
    signal = np.concatenate([packet.data for packet in packets if packet.id==id], axis=0)
    
    #izracunaj cas
    lengthSignal = signal.shape[0]
    time = (timestamps[-1] - timestamps[0])/1000.0

    #frekvneca signala
    Fvz = lengthSignal/time
    return signal, Fvz

def obdelaj_vse(surova_mapa: str, obdelana_mapa: str):
    """Za implementacijo preberi README.md"""
    for razred in RAZREDI:
        vhodna = Path(surova_mapa) / razred
        izhodna = Path(obdelana_mapa) / razred
        izhodna.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Obdelujem razred: {razred} ===")

        for pot in sorted(vhodna.glob("*.BIN")):
            if not pot.is_file():
                continue

            print(f"\nObdelujem: {pot} ({pot.stat().st_size} bajtov)")

            with open(pot, "rb") as f:
                data = f.read()

            # zelo majhne datoteke skoraj zagotovo niso pravi log
            if len(data) < 20:
                print(f"SKIP: datoteka je premajhna: {pot}")
                continue

            # če je notri tekst ERROR, to ni pravi BIN log
            if b"ERROR" in data[:200]:
                print(f"SKIP: datoteka vsebuje ERROR tekst: {pot}")
                print(data[:200].decode(errors="replace"))
                continue

            global id, chunks, timestamps
            id, chunks, timestamps = [], [], []

            try:
                separate(data)
            except Exception as e:
                print(f"SKIP: napaka pri parsiranju datoteke {pot.name}")
                print(f"Napaka: {type(e).__name__}: {e}")
                continue

            if len(chunks) == 0:
                print(f"SKIP: v datoteki ni uporabnih chunkov: {pot.name}")
                continue

            id_arr = np.array(id)
            ts_arr = np.array(timestamps)
            ch_arr = np.array(chunks, dtype=object)
            packets = []

            for pid, ptimestamp, pchunk in zip(id_arr, ts_arr, ch_arr):
                pdata = np.array(pchunk, dtype=np.int16).flatten()
                packets.append(Paket(pid, ptimestamp, pdata))

            try:
                signal, Fvz = sestavi_podatke(packets)
            except Exception as e:
                print(f"SKIP: ne morem sestaviti signala za {pot.name}")
                print(f"Napaka: {type(e).__name__}: {e}")
                continue

            Fvz = Fvz * 0.18
            Fvz = round(Fvz)

            signal = signal[Fvz:]

            for i in range(len(signal)):
                if signal[i] < 0:
                    signal[i] = -ALAW_DECODE_TABLE[-signal[i]]
                else:
                    signal[i] = ALAW_DECODE_TABLE[signal[i]]

            start, end = najdi_dogodek(signal, Fvz)

            spektogram = signal_v_spektogram(signal, start, end, Fvz, normalize_16bit=True)
            np.save(izhodna / (pot.name + ".npy"), spektogram)
            np.save(izhodna / (pot.name + "_freq.npy"), freq_mask(spektogram))

            for aug_signal, suffix in augmentiraj_podatke(signal, Fvz):
                aug_spec = signal_v_spektogram(aug_signal, start, end, Fvz, normalize_16bit=True)
                np.save(izhodna / (pot.name + suffix + ".npy"), aug_spec)
                np.save(izhodna / (pot.name + suffix + "_freq.npy"), freq_mask(aug_spec))

def freq_mask(spec: np.ndarray, F: int = 10) -> np.ndarray:
    spec = spec.copy()
    f = np.random.randint(1, F + 1)
    f0 = np.random.randint(0, spec.shape[0] - f)
    spec[f0:f0 + f, :] = 0
    return spec

def spectogram_norm(signal: np.ndarray) -> np.ndarray:
    Sxx_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    return (Sxx_norm * 255).astype(np.float32)
    

def shrani_spektrogram(spektogram: np.ndarray):
    spektogram_norm = spectogram_norm(spektogram)
    np.save("odpadki_obdelani\\papir\\karton\\karton_skatla3.npy", spektogram_norm)

def signal_v_spektogram(signal: np.ndarray, startInd: int, endInd: int, Fvz: float, normalize_16bit: bool = False):
    interval = signal[startInd:endInd]
    time = np.arange(startInd, endInd) / Fvz

    if normalize_16bit:
        sig_float = interval.astype(np.float64)
        peak = np.max(sig_float)
        low = np.min(sig_float)
        if peak > 0:
            interval = (sig_float - low)/(peak - low) * 65535 - 32768
        interval = interval.astype(np.int16)

    spec_sig = np.array(interval, dtype=np.float64)
        
    nperseg = min(256, max(16, len(spec_sig) // 8))
    f_spec, t_spec, Sxx = sp_signal.spectrogram(spec_sig, fs=Fvz, nperseg=nperseg)

    spectogram_input = 10 * np.log10(Sxx + 1e-10)

    spectogram_input = spectogram_norm(spectogram_input)

    return spectogram_input

def pripravi_test_podatke(obdelana_mapa: str, test_mapa: str, delez: float = 0.1):
    for razred in RAZREDI:
        vhodna = Path(obdelana_mapa) / razred
        izhodna = Path(test_mapa) / razred
        izhodna.mkdir(parents=True, exist_ok=True)

        # izberemo samo originalne posnetke (ne augmentirane)
        baze = sorted(p for p in vhodna.glob("*.npy") if p.name.endswith(".BIN.npy"))
        n = max(1, int(len(baze) * delez))
        izbrane_baze = random.sample(baze, min(n, len(baze)))

        kopirano = 0
        for baza in izbrane_baze:
            # kopiramo original + vse augmentirane verzije tega posnetka
            for pot in vhodna.glob(f"{baza.stem}*.npy"):
                shutil.copy2(pot, izhodna / pot.name)
                kopirano += 1

        print(f"{razred}: {len(izbrane_baze)}/{len(baze)} posnetkov ({kopirano} datotek) -> test_data/{razred}")

if __name__ == "__main__":


   obdelaj_vse("Audio_logs", "odpadki_obdelani")
   pripravi_test_podatke("odpadki_obdelani", "test_data")
   """with open("odpadki_surovi\\papir\\karton\\karton_skatla3", "rb") as f:
    data = f.read()
    separate(data)

    id = np.array(id)
    timestamp = np.array(timestamps)
    chunks = np.array(chunks, dtype=object)
    packets = []
    

    for id, timestamp, chunk in zip(id, timestamps, chunks):
        data = np.array(chunk, dtype=np.int16).flatten()
        packets.append(Paket(id, timestamp, data))

    signal, Fvz = sestavi_podatke(packets)

    for i in range(len(signal)):
        if signal[i] < 0:
            signal[i] = -ALAW_DECODE_TABLE[-signal[i]]
        else:
            signal[i] = ALAW_DECODE_TABLE[signal[i]]

    start, end = najdi_dogodek(signal, Fvz)
    prikazi_signal(signal, "Microphone", None, None, Fvz)
    prikazi_signal(signal, "Microphone — event only", start, end, Fvz, normalize_16bit=True)"""
   
    

    