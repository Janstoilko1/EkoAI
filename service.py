import serial
import socket
import threading
import serial.tools.list_ports
from pathlib import Path
import time

SERVICE_PORT = 5000
HOST = "127.0.0.1"

STM32_PORT = "COM3"
BAUDRATE = 115200

PID = 0x5740
VID = 0x0483

"""
SPREMENI OUTPUT_FOLDER na pot, kamor želiš shranjevati audio loge, ki jih dobiš iz STM32.

"""

OUTPUT_FOLDER = Path(r"C:\Users\Jan\OneDrive - Univerza v Mariboru\Desktop\2. letnik\SPO\zapiski\saved_files") 
OUTPUT_FOLDER.mkdir(exist_ok=True)
# Koliko sekund po zadnjem prejetem bajtu še čakamo
END_TIMEOUT = 2.0

def find_stm32_port():
    for port in serial.tools.list_ports.comports():
        if port.vid == VID and port.pid == PID and port.device == STM32_PORT:
            print(f"Found STM32 on {port.device}")
            return port.device

    return None

def save_audio_log(fileData, outputFile):
    with open(outputFile, "wb") as f:
        f.write(fileData)

def read_log_from_stm(ser, command):
    ser.reset_input_buffer()

    print(f"POŠILJAM UKAZ....{command.strip()}")
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

def parse_list_response(data):
    text = data.decode("ascii", errors="replace")

    files = []

    for line in text.splitlines():
        line = line.strip()

        if line == "":
            continue

        if line == "Listing files...":
            continue

        if line == "Volume is FAT32":
            continue

        if line == ">":
            continue

        parts = line.split()

        if len(parts) >= 2 and parts[0].upper().endswith(".BIN"):
            filename = parts[0]
             #size = int(parts[1])

            files.append(filename)

    return files

def get_all_logs():
    with serial.Serial(STM32_PORT, BAUDRATE, timeout=0.2) as ser:
        time.sleep(2)  # malo počaka, da se STM/USB serial stabilizira
        #DOBIMO LIST VSEH AUDIO LOGOV
        data = read_log_from_stm(ser, command="LIST\r\n")

        files = parse_list_response(data)

        for i,file in enumerate(files):
            fileData = read_log_from_stm(ser, command=f"GET {file}\r\n")

            outputFile = OUTPUT_FOLDER/f"Audio_log{i+1}" 

            save_audio_log(fileData,outputFile) 

def get_last_log():
    with serial.Serial(STM32_PORT, BAUDRATE, timeout=0.2) as ser:
        time.sleep(2)  # malo počaka, da se STM/USB serial stabilizira
        #DOBIMO LIST VSEH AUDIO LOGOV
        data = read_log_from_stm(ser, command="LIST\r\n")

        files = parse_list_response(data)
        time.sleep(2)

        lenght = len(files)

        file = files[-1]

        fileData = read_log_from_stm(ser, command=f"GET {file}\r\n")

        outputFile = OUTPUT_FOLDER/f"Audio_log{lenght}" 

        save_audio_log(fileData,outputFile)

def get_file_by_name(commandByUser):
    with serial.Serial(STM32_PORT, BAUDRATE, timeout=0.2) as ser:
        time.sleep(2)  # malo počaka, da se STM/USB serial stabilizira
        #DOBIMO LIST VSEH AUDIO LOGOV

        index = commandByUser.find("|")
        name = commandByUser[index+1:]

        fileData = read_log_from_stm(ser, command=f"GET {name}\r\n")
        outputFile = OUTPUT_FOLDER/f"{name}"

        save_audio_log(fileData, outputFile)
        
def delete_files():
    with serial.Serial(STM32_PORT, BAUDRATE, timeout=0.2) as ser:
        time.sleep(2)  # malo počaka, da se STM/USB serial stabilizira
        #DOBIMO LIST VSEH AUDIO LOGOV

        fileData = read_log_from_stm(ser, command=f"DELETE\r\n")
def listen_for_recording():
    with serial.Serial(STM32_PORT, BAUDRATE, timeout=0.2) as ser:
        data = bytearray()

        while True:
            chunk = ser.readline()

            if chunk.decode(errors="replace").strip() == "Logging stopped.":
                time.sleep(3)  # počakamo malo, da se STM32 stabilizira
                data = read_log_from_stm(ser, command="LIST\r\n")

                files = parse_list_response(data)
                time.sleep(5)

                lenght = len(files)

                file = files[-1]

                fileData = read_log_from_stm(ser, command=f"GET {file}\r\n")

                outputFile = OUTPUT_FOLDER/f"Audio_log{lenght}" 

                save_audio_log(fileData,outputFile)

                

            
        
def handle_client(conn, addr):
    with conn:
        stm32_port = find_stm32_port()

        buffer = ""
        

        if stm32_port is not None:
            conn.sendall(f'Connected to SPO STM32 service - STM32 detected at {stm32_port}\n'.encode())

        else:
            conn.sendall(b"Connected to SPO STM32 service - No STM32 detected\n")

        while True:
            data = conn.recv(1024)

            if not data:
                break
                
            buffer  += data.decode(errors="replace")

            while "\n" in buffer:
                line, buffer = buffer.split("\n",1)

                command = line.strip()

                print(command)
                

                if command == "":
                    continue

                if command == 'STATUS':
                    stm32_port = find_stm32_port()

                    if stm32_port is not None:
                        response = "STM32 is connected\n"
                    else:
                        response = "STM32 is not connected\n"

                elif command == "GET_ALL":
                    stm32_port = find_stm32_port()

                    if stm32_port is None:
                        response = "FAIL: STM32 is not connected\n"
                    else:
                        response = "GETTING ALL AUDIO LOGS...."
                        get_all_logs()
                
                elif command == "GET_LAST":
                    stm32_port = find_stm32_port()

                    if stm32_port is None:
                        response = "FAIL: STM32 is not connected\n"
                    else:
                        response = "GETTING LATEST AUDIO LOG...."
                        get_last_log()

                elif command.startswith("GET_FILE|"):
                    stm32_port = find_stm32_port()

                    if stm32_port is None:
                        response = "FAIL: STM32 is not connected\n"
                    else:
                        response = "GETTING FILE...."
                        get_file_by_name(command)
                elif command == "DELETE":
                    stm32_port = find_stm32_port()

                    if stm32_port is None:
                        response = "FAIL: STM32 is not connected\n"
                    else:
                        response = "DELETING FILES...."
                        delete_files()
                elif command == "LISTEN":
                    stm32_port = find_stm32_port()

                    if stm32_port is None:
                        response = "FAIL: STM32 is not connected\n"
                    else:
                        response = "LISTENING...."
                        listen_for_recording()

                else:
                    response = f"FAIL: Unknown command: {command}\n"

                conn.sendall(response.encode())

def main():
     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, SERVICE_PORT))
        server.listen()

        print(f"Service listening on {HOST}:{SERVICE_PORT}")

        while True:
            # conn je povezava do klienta, preko tega lahko beremo in pošiljamo podatke
            # addr je tuple nalova ('127.0.0.1, 54321)
            conn, addr = server.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            thread.start()

if __name__ == '__main__':
    main()



