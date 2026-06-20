import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import struct
from pathlib import Path
from scipy.io.wavfile import write
import tempfile

st.set_page_config(layout="wide")

st.title("BIN Audio Viewer")

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
  25088,26112,27136,28160,29184,30208,31232,32256
], dtype=np.int16)

folder_path = st.sidebar.text_input(
    "Pot do mape z BIN datotekami",
    value="."
)

folder = Path(folder_path)

if not folder.exists():
    st.error("Mapa ne obstaja.")
    st.stop()

bin_files = [
    f for f in folder.rglob("*")
    if f.is_file()
]

if len(bin_files) == 0:
    st.warning("Ni datotek.")
    st.stop()

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

def unstuff(byteData, size):

    unstuffed = []

    i = 0

    while i < size:

        if i <= size - 2:

            if byteData[i] == 0xFE:

                unstuffed.append(0xFE ^ byteData[i + 1])

                i += 2

                continue

        unstuffed.append(byteData[i])

        i += 1

    return bytes(unstuffed)

def load_bin_audio(filepath):

    with open(filepath, "rb") as f:
        data = f.read()

    chunks = []
    timestamps = []

    start = 0
    stop = 0

    while (
        start < len(data)
        and data[start:start + 2] != b'\xFF\xFF'
    ):
        start += 1

    stop = start + 1

    while stop < len(data):

        while (
            stop < len(data)
            and data[stop:stop + 2] != b'\xFF\xFF'
        ):
            stop += 1

        packet = data[start + 2:stop]

        try:

            payload = unstuff(
                packet[1:],
                len(packet[1:])
            )

            timestamp = struct.unpack(
                '<I',
                payload[0:4]
            )[0]

            packet_size = struct.unpack(
                '<H',
                payload[4:6]
            )[0] + 1

            crc = struct.unpack(
                '<H',
                payload[
                    6 + packet_size:
                    6 + packet_size + 2
                ]
            )[0]

            computed_crc = compute_crc(
                payload[:6 + packet_size]
            )

            if crc != computed_crc:

                start = stop
                stop = start + 1
                continue

            chunks_data = payload[6:6 + packet_size]

            pos = 0

            while pos < len(chunks_data):

                chunk_id = chunks_data[pos]

                chunk_size_enc = struct.unpack(
                    '<H',
                    chunks_data[pos + 1:pos + 3]
                )[0]

                chunk_size = chunk_size_enc + 1

                chunk_data = chunks_data[
                    pos + 4:pos + 4 + chunk_size
                ]

                if chunk_id == 4:

                    for i in range(chunk_size):

                        try:

                            value = struct.unpack(
                                '<b',
                                chunk_data[i:i + 1]
                            )[0]

                            chunks.append(value)

                        except:
                            break

                    timestamps.append(timestamp)

                pos += 4 + chunk_size

        except:
            pass

        start = stop

        while (
            start < len(data)
            and data[start:start + 2] != b'\xFF\xFF'
        ):
            start += 1

        stop = start + 1

    if len(chunks) == 0:
        return None, None

    signal = np.array(chunks)

    # decode ALAW
    decoded = np.zeros_like(signal)

    for i in range(len(signal)):

        if signal[i] < 0:
            decoded[i] = -ALAW_DECODE_TABLE[-signal[i]]
        else:
            decoded[i] = ALAW_DECODE_TABLE[signal[i]]

    timestamps = np.array(timestamps)

    if len(timestamps) < 2:
        return None, None

    total_time = (
        timestamps[-1] - timestamps[0]
    ) / 1000.0

    Fvz = len(decoded) / total_time

    return decoded.astype(np.int16), int(Fvz)

for file in bin_files:

    st.divider()

    st.subheader(file.name)

    signal, Fvz = load_bin_audio(file)

    if signal is None:

        st.error("Napaka pri branju.")
        continue


    tmp_wav = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    )

    write(
        tmp_wav.name,
        Fvz,
        signal
    )

    st.audio(tmp_wav.name)

    fig, ax = plt.subplots(figsize=(14, 4))

    time = np.arange(len(signal)) / Fvz

    ax.plot(time, signal)

    ax.set_title(file.name)

    ax.set_xlabel("Time [s]")

    ax.set_ylabel("Amplitude")

    ax.grid(True)

    st.pyplot(fig)