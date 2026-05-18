import serial
import time
from pathlib import Path

# NASTAVI TO
PORT = "COM3"          # zamenjaj s svojim COM portom
BAUDRATE = 9600     # zamenjaj, če imaš drugo hitrost

OUTPUT_FOLDER = Path(r"C:\Users\Jan\OneDrive - Univerza v Mariboru\Desktop\EkoAI\Audio_logs\steklo")
OUTPUT_FOLDER.mkdir(exist_ok=True)

FIRST_LOG = 10
LAST_LOG = 31

# Tukaj nastavi ukaz, ki ga tvoj STM32 pričakuje.
# Poskusi eno od variant:
GET_COMMAND_TEMPLATE = "GET LOG0{num}.BIN\r\n"
# GET_COMMAND_TEMPLATE = "GET LOG{num}\r\n"
# GET_COMMAND_TEMPLATE = "GET {num}\r\n"

# Koliko sekund po zadnjem prejetem bajtu še čakamo
END_TIMEOUT = 2.0


def read_log_from_stm(ser, command):
    ser.reset_input_buffer()

    print(f"Pošiljam ukaz: {command.strip()}")
    ser.write(command.encode("ascii"))
    ser.flush()

    data = bytearray()
    last_data_time = time.time()

    while True:
        chunk = ser.read(4096)

        if chunk:
            data.extend(chunk)
            last_data_time = time.time()
        else:
            if time.time() - last_data_time > END_TIMEOUT:
                break

    return bytes(data)


with serial.Serial(PORT, BAUDRATE, timeout=0.2) as ser:
    time.sleep(2)  # malo počaka, da se STM/USB serial stabilizira

    for num in range(FIRST_LOG, LAST_LOG + 1):
        command = GET_COMMAND_TEMPLATE.format(num=num)
        data = read_log_from_stm(ser, command)

        output_file = OUTPUT_FOLDER / f"LOG0{num}.BIN"

        with open(output_file, "wb") as f:
            f.write(data)

        print(f"Shranjeno: {output_file}  ({len(data)} bajtov)")

print("Končano.")