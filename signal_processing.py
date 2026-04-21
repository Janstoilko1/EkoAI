import struct
from attr import dataclass
import numpy as np
import matplotlib.pyplot as plt
import string

@dataclass
class Paket:
	id: int      
	time: float    
	data: np.ndarray

ALAW_DECODE_TABLE = np.array([           8,    24,    40,    56,    72,    88,   104,   120,   136,         152,   168,   184,   200,   216,   232,   248,   264,   280,         296,   312,   328,   344,   360,   376,   392,   408,   424,         440,   456,   472,   488,   504,   528,   560,   592,   624,         656,   688,   720,   752,   784,   816,   848,   880,   912,         944,   976,  1008,  1056,  1120,  1184,  1248,  1312,  1376,        1440,  1504,  1568,  1632,  1696,  1760,  1824,  1888,  1952,        2016,  2112,  2240,  2368,  2496,  2624,  2752,  2880,  3008,        3136,  3264,  3392,  3520,  3648,  3776,  3904,  4032,  4224,        4480,  4736,  4992,  5248,  5504,  5760,  6016,  6272,  6528,        6784,  7040,  7296,  7552,  7808,  8064,  8448,  8960,  9472,        9984, 10496, 11008, 11520, 12032, 12544, 13056, 13568, 14080,       14592, 15104, 15616, 16128, 16896, 17920, 18944, 19968, 20992,       22016, 23040, 24064, 25088, 26112, 27136, 28160, 29184, 30208,       31232, 32256], dtype=np.int16)

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
    print("crc ok")

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

        # Parse sensor samples (3x int16_t per sample or 1x int16_t)
        samples = []
        if chunk_id == 1 or chunk_id == 2 or chunk_id == 3:
            for i in range(0, chunk_size, 6):
                x, y, z = struct.unpack('<hhh', chunk_data[i:i+6])
                samples.append((x, y, z))
        elif chunk_id == 4:
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


#grafi
     
def normalize (vektor: np.ndarray, id:int) -> np.ndarray:
    if id == 1:
        return vektor*8.75e-3
    if id == 2:
        return vektor*6.125e-5
    if id == 3:
        return vektor*1.5e-3
    return vektor


def prikazi_signal (signal: np.ndarray, naslov: string, startInd: int, endInd: int, id: int):
    if startInd is None:
        startInd = 0
    if endInd is None:
        endInd = len(signal)

    interval = signal[startInd:endInd]
    time = np.arange(startInd, endInd) / Fvz

    plt.figure()
    if id != 4:
        plt.plot(time,[num[0] for num in interval], label="X")
        plt.plot(time, [num[1] for num in interval], label="Y")
        plt.plot(time, [num[2] for num in interval], label="Z")
    else:
        plt.plot(time, interval, label="Value")
    plt.xlabel("time [s]")
    plt.ylabel("Amplitude")
    plt.title(naslov)
    plt.legend()
    plt.show()

def sestavi_podatke (packets: np.ndarray, id:int): 
    timestamps = [packet.time for packet in packets if packet.id==id]
    signal = normalize(np.concatenate([packet.data for packet in packets if packet.id==id], axis=0), id)
    
    #izracunaj cas
    lengthSignal = signal.shape[0]
    time = (timestamps[-1] - timestamps[0])/1000.0

    #frekvneca signala
    Fvz = lengthSignal/time
    return signal, Fvz

if __name__ == "__main__":
    with open("Audio_logs/LOG6", "rb") as f:
        data = f.read()
        separate(data)

    id = np.array(id)
    timestamp =  np.array(timestamps)
    chunks = np.array(chunks, dtype=object)
    graph_id = 4
    packets=[]

    for id,timestamp,chunk in zip(id,timestamps,chunks):
        data = np.array(chunk,dtype=np.int16).flatten()
        if id == 1 or id == 2 or id == 3:   
            data = data.reshape(-1,3)
        packets.append(Paket(id,timestamp,data))

    signal, Fvz= sestavi_podatke(packets,graph_id)

    if graph_id == 4:
        for i in range(len(signal)):
            if signal[i] < 0:
                signal[i] = -ALAW_DECODE_TABLE[-signal[i]]
            else:
                signal[i] = ALAW_DECODE_TABLE[signal[i]]

    prikazi_signal(signal, "Microphone", None, None, graph_id)
    prikazi_signal(signal, "Microphone zoomed", int(Fvz*0.7), int(Fvz*1.3), graph_id)