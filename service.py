import serial
import socket
import threading
import queue
import serial.tools.list_ports
from pathlib import Path
import time
import os
import wave
import tempfile
import logging

import numpy as np

try:
    import winsound  # Predvajanje zvoka (samo Windows)
except ImportError:
    winsound = None

from prediction_utils import WastePredictor

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =========================
# NASTAVITVE
# =========================

SERVICE_PORT = 5000
HOST = "127.0.0.1"

BAUDRATE = 115200

PID = 0x5740
VID = 0x0483

OUTPUT_FOLDER = Path(
    r"C:\Users\Jan\OneDrive - Univerza v Mariboru\Desktop\2. letnik\SPO\zapiski\saved_files"
)

END_TIMEOUT = 2.0
SERIAL_TIMEOUT = 0.2

# Če želiš zakleniti samo na COM3, nastavi npr. "COM3".
# Če želiš, da najde STM32 ne glede na COM številko, pusti None.
PREFERRED_STM32_PORT = None


#MODEL
predictor = WastePredictor("model9.pth")
prediction_lock = threading.Lock()

# =========================
# GLOBALNO STANJE
# =========================

serial_lock = threading.Lock()

listen_thread = None
listen_stop_event = threading.Event()

is_predicted = False

# Vrsta za posredovanje rezultatov iz delovnih niti v GUI (glavna nit).
gui_queue = queue.Queue()

# =========================
# LOGGING
# =========================

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_FOLDER / "spo_stm32_service.log", encoding="utf-8")
    ]
)


# =========================
# STM32 ISKANJE
# =========================

def find_stm32_port():
    """
    Poišče STM32 po VID/PID.
    Vrne npr. 'COM3' ali None.
    """

    try:
        for port in serial.tools.list_ports.comports():
            if port.vid == VID and port.pid == PID:
                if PREFERRED_STM32_PORT is not None:
                    if port.device != PREFERRED_STM32_PORT:
                        continue

                logging.info(f"Found STM32 on {port.device}")
                return port.device

    except Exception as e:
        logging.error(f"Napaka pri iskanju STM32: {e}")

    return None


def is_stm32_connected():
    return find_stm32_port() is not None


# =========================
# SHRANJEVANJE
# =========================

def safe_filename(filename):
    """
    Prepreči čudne poti, npr. ../../nekaj.
    """

    filename = os.path.basename(filename.strip())

    if filename == "":
        return None

    return filename


def save_audio_log(file_data, output_file):
    """
    Varno shrani datoteko.
    """

    try:
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

        with open(output_file, "wb") as f:
            f.write(file_data)

        logging.info(f"Shranjeno: {output_file} ({len(file_data)} bajtov)")
        return True, f"Saved {output_file.name}\n"

    except PermissionError:
        return False, f"FAIL: Permission denied while saving {output_file}\n"

    except OSError as e:
        return False, f"FAIL: Could not save file: {e}\n"

    except Exception as e:
        return False, f"FAIL: Unexpected save error: {e}\n"


# =========================
# SERIAL BRANJE/PISANJE
# =========================

def open_stm32_serial():
    """
    Odpre serial povezavo do STM32.
    """

    stm32_port = find_stm32_port()

    if stm32_port is None:
        return None, "FAIL: STM32 is not connected\n"

    try:
        ser = serial.Serial(
            stm32_port,
            BAUDRATE,
            timeout=SERIAL_TIMEOUT,
            write_timeout=2
        )

        time.sleep(1)
        return ser, None

    except serial.SerialException as e:
        return None, f"FAIL: Could not open STM32 serial port: {e}\n"

    except OSError as e:
        return None, f"FAIL: OS error while opening STM32: {e}\n"

    except Exception as e:
        return None, f"FAIL: Unexpected serial open error: {e}\n"


def read_text_command_from_stm(ser, command):
    """
    Za tekstovne ukaze kot LIST in DELETE.
    Bere do znaka '>' ali do timeouta.
    """

    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        logging.info(f"POŠILJAM UKAZ: {command.strip()}")

        ser.write(command.encode("ascii"))
        ser.flush()

        data = bytearray()
        last_data_time = time.time()

        while True:
            chunk = ser.read(4096)

            if chunk:
                data.extend(chunk)
                last_data_time = time.time()

                if b">" in data:
                    break

            else:
                if time.time() - last_data_time > END_TIMEOUT:
                    break

        return True, bytes(data)

    except serial.SerialException as e:
        return False, f"FAIL: Serial error: {e}\n".encode()

    except OSError as e:
        return False, f"FAIL: OS error: {e}\n".encode()

    except Exception as e:
        return False, f"FAIL: Unexpected serial error: {e}\n".encode()


def read_file_from_stm(ser, filename, expected_size=None):
    """
    Prebere binarno datoteko iz STM32.
    Če poznamo velikost iz LIST, prebere točno toliko bajtov.
    """

    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        command = f"GET {filename}\r\n"

        logging.info(f"POŠILJAM UKAZ: {command.strip()}")

        ser.write(command.encode("ascii"))
        ser.flush()

        data = bytearray()
        last_data_time = time.time()

        if expected_size is not None:
            file_start = -1

            while True:
                if file_start != -1 and len(data) - file_start >= expected_size:
                    break

                chunk = ser.read(4096)

                if chunk:
                    data.extend(chunk)
                    last_data_time = time.time()

                    if file_start == -1:
                        file_start = data.find(b"\xFF\xFF")
                else:
                    if time.time() - last_data_time > END_TIMEOUT:
                        break

            if file_start == -1:
                return False, (
                    f"FAIL: Could not find binary packet start in {filename}. "
                    f"Received {len(data)} bytes\n"
                ).encode()

            received_size = len(data) - file_start

            if received_size < expected_size:
                return False, (
                    f"FAIL: Timeout while reading {filename}. "
                    f"Expected {expected_size} binary bytes, got {received_size} bytes\n"
                ).encode()

            skipped = file_start

            if skipped > 0:
                logging.info(f"Skipped {skipped} non-binary bytes before {filename}")

            return True, bytes(data[file_start:file_start + expected_size])

        else:
            while True:
                chunk = ser.read(4096)

                if chunk:
                    data.extend(chunk)
                    last_data_time = time.time()
                else:
                    if time.time() - last_data_time > END_TIMEOUT:
                        break

            if len(data) == 0:
                return False, f"FAIL: No data received for {filename}\n".encode()

            return True, bytes(data)

    except serial.SerialException as e:
        return False, f"FAIL: Serial error while reading file: {e}\n".encode()

    except OSError as e:
        return False, f"FAIL: OS error while reading file: {e}\n".encode()

    except Exception as e:
        return False, f"FAIL: Unexpected file read error: {e}\n".encode()


# =========================
# LIST PARSER
# =========================

def parse_list_response(data):
    """
    Iz LIST odgovora naredi seznam:
    [
        ("LOG001.BIN", 27182),
        ("LOG002.BIN", 18390),
        ...
    ]
    """

    try:
        text = data.decode("ascii", errors="replace")

        files = []

        for line in text.splitlines():
            line = line.strip()

            if line == "":
                continue

            if line.startswith("Listing files"):
                continue

            if line == "Volume is FAT32":
                continue

            if line == ">":
                continue

            parts = line.split()

            if len(parts) >= 2 and parts[0].upper().endswith(".BIN"):
                filename = parts[0]

                try:
                    size = int(parts[1])
                except ValueError:
                    size = None

                files.append((filename, size))

        return files

    except Exception as e:
        logging.error(f"Napaka pri parsiranju LIST odgovora: {e}")
        return []


def get_file_list_from_open_serial(ser):
    """
    Pošlje LIST in vrne seznam datotek.
    """

    ok, data = read_text_command_from_stm(ser, "LIST\r\n")

    if not ok:
        return False, data.decode(errors="replace"), []

    files = parse_list_response(data)

    if len(files) == 0:
        logging.warning("LIST ni vrnil nobenih datotek.")
        logging.warning("Raw LIST response:")
        logging.warning(data.decode("ascii", errors="replace"))

    return True, data.decode("ascii", errors="replace"), files


# =========================
# UKAZI ZA STM32
# =========================

def get_all_logs():
    """
    TCP ukaz GET_ALL:
    1. pošlje LIST
    2. dobi vse datoteke
    3. za vsako naredi GET filename
    4. shrani na disk
    """

    if listen_thread is not None and listen_thread.is_alive():
        return "FAIL: Stop LISTEN first with STOP_LISTEN\n"

    with serial_lock:
        ser, error = open_stm32_serial()

        if ser is None:
            return error

        try:
            with ser:
                ok, list_text, files = get_file_list_from_open_serial(ser)

                if not ok:
                    return list_text

                if len(files) == 0:
                    return "FAIL: No files found on STM32\n"

                saved_count = 0

                for filename, size in files:
                    safe_name = safe_filename(filename)

                    if safe_name is None:
                        logging.warning(f"Skipping invalid filename: {filename}")
                        continue

                    ok, file_data = read_file_from_stm(ser, safe_name, size)

                    if not ok:
                        logging.error(file_data.decode(errors="replace").strip())
                        continue

                    output_file = OUTPUT_FOLDER / safe_name

                    ok, message = save_audio_log(file_data, output_file)

                    if ok:
                        saved_count += 1
                    else:
                        logging.error(message.strip())

                return f"All files from STM32 are processed. Saved {saved_count}/{len(files)} files\n"

        except Exception as e:
            return f"FAIL: GET_ALL failed: {e}\n"


def get_last_log_from_open_serial(ser):
    """
    Prenese zadnjo datoteko iz že odprtega serial porta.
    To se uporablja tudi v LISTEN.
    """

    ok, list_text, files = get_file_list_from_open_serial(ser)

    if not ok:
        return list_text

    if len(files) == 0:
        return "FAIL: No files found on STM32\n"

    filename, size = files[-1]

    safe_name = safe_filename(filename)

    if safe_name is None:
        return "FAIL: Invalid filename from STM32\n"

    ok, file_data = read_file_from_stm(ser, safe_name, size)

    logging.info(f"fileData size: {len(file_data)} bytes")

    print(len(file_data), size)

    if not ok:
        return file_data.decode(errors="replace")

    output_file = OUTPUT_FOLDER / safe_name

    ok, message = save_audio_log(file_data, output_file)

    if not ok:
        return message

    return "Last file from STM32 has been processed\n", file_data


def get_last_log():
    """
    TCP ukaz GET_LAST.
    """

    if listen_thread is not None and listen_thread.is_alive():
        return "FAIL: Stop LISTEN first with STOP_LISTEN\n"

    with serial_lock:
        ser, error = open_stm32_serial()

        if ser is None:
            return error

        try:
            with ser:
                message, file_data = get_last_log_from_open_serial(ser)
                return message
        except Exception as e:
            return f"FAIL: GET_LAST failed: {e}\n"


def get_file_by_name(command_by_user):
    """
    TCP ukaz:
    GET_FILE|LOG001.BIN
    """

    if listen_thread is not None and listen_thread.is_alive():
        return "FAIL: Stop LISTEN first with STOP_LISTEN\n"

    index = command_by_user.find("|")

    if index == -1:
        return "FAIL: Missing '|'. Use GET_FILE|filename\n"

    filename = command_by_user[index + 1:].strip()
    filename = safe_filename(filename)

    if filename is None:
        return "FAIL: Invalid filename\n"

    with serial_lock:
        ser, error = open_stm32_serial()

        if ser is None:
            return error

        try:
            with ser:
                # Najprej naredimo LIST, da dobimo velikost datoteke.
                ok, list_text, files = get_file_list_from_open_serial(ser)

                if not ok:
                    return list_text

                expected_size = None

                for file_name, size in files:
                    if file_name.upper() == filename.upper():
                        expected_size = size
                        break

                ok, file_data = read_file_from_stm(ser, filename, expected_size)

                if not ok:
                    return file_data.decode(errors="replace")

                output_file = OUTPUT_FOLDER / filename

                ok, message = save_audio_log(file_data, output_file)

                if not ok:
                    return message

                return f"File {filename} from STM32 has been processed\n"

        except Exception as e:
            return f"FAIL: GET_FILE failed: {e}\n"


def delete_files():
    """
    TCP ukaz DELETE.
    """

    if listen_thread is not None and listen_thread.is_alive():
        return "FAIL: Stop LISTEN first with STOP_LISTEN\n"

    with serial_lock:
        ser, error = open_stm32_serial()

        if ser is None:
            return error

        try:
            with ser:
                ok, data = read_text_command_from_stm(ser, "DELETE\r\n")

                if not ok:
                    return data.decode(errors="replace")

                logging.info(data.decode("ascii", errors="replace"))

                return "All files on STM32 are deleted\n"

        except Exception as e:
            return f"FAIL: DELETE failed: {e}\n"


# =========================
# LISTEN FUNKCIJE
# =========================

def listen_for_recording(stop_event):
    """
    Posluša STM32.
    Ko dobi 'Logging stopped.', shrani zadnji log.
    Ustavi se, ko stop_event postane True.
    """

    stm32_port = find_stm32_port()

    if stm32_port is None:
        logging.error("FAIL: STM32 is not connected")
        return

    try:
        with serial_lock:
            with serial.Serial(
                stm32_port,
                BAUDRATE,
                timeout=SERIAL_TIMEOUT,
                write_timeout=2
            ) as ser:

                time.sleep(1)
                logging.info("LISTEN started")

                while not stop_event.is_set():
                    try:
                        chunk = ser.readline()

                        if not chunk:
                            continue

                        line = chunk.decode(errors="replace").strip()

                        if line:
                            logging.info(f"STM32: {line}")

                        if line == "Logging stopped.":
                            logging.info("Recording stopped detected. Waiting before reading last log...")
                            time.sleep(3)

                            message, fileData = get_last_log_from_open_serial(ser)
                            logging.info(message.strip())

                            """GLAVNA FUNKCIJA SAMEGA PREDICTIONA, ZA ZDAJ ŠE NI POVEZANO Z GUI, AMPAK OD TAM
                                KLIČE FUNKCIJO OBDELAJ.
                                
                                RETURNA MESSAGE, KATERI BO VSEBOVAL SAM PREDICTION"""
                            

                            prediction_message = make_prediction(fileData)

                            logging.info(prediction_message.strip())
                        

                    except serial.SerialException as e:
                        logging.error(f"FAIL: Serial error while listening: {e}")
                        break

                    except OSError as e:
                        logging.error(f"FAIL: OS error while listening: {e}")
                        break

                    except Exception as e:
                        logging.error(f"FAIL: Unexpected error while listening: {e}")
                        continue

    finally:
        logging.info("LISTEN stopped")


def start_listen():
    """
    Zažene LISTEN v ozadju.
    """

    global listen_thread

    if listen_thread is not None and listen_thread.is_alive():
        return "FAIL: LISTEN is already running\n"

    if find_stm32_port() is None:
        return "FAIL: STM32 is not connected\n"

    listen_stop_event.clear()

    listen_thread = threading.Thread(
        target=listen_for_recording,
        args=(listen_stop_event,),
        daemon=True
    )

    listen_thread.start()

    return "LISTENING started\n"


def stop_listen():
    """
    Ustavi LISTEN.
    """

    global listen_thread

    if listen_thread is None or not listen_thread.is_alive():
        return "LISTEN is not running\n"

    listen_stop_event.set()
    listen_thread.join(timeout=5)

    if listen_thread.is_alive():
        return "FAIL: LISTEN did not stop yet\n"

    return "LISTENING stopped\n"

def make_prediction(file_data):
    try:
        with prediction_lock:
            result = predictor.predict_from_bytes(file_data)
        
        razred = result["razred"]
        podrazred = result["podrazred"]
        confidence = result["confidence"]

        message = (
            f"PREDICTION: {razred} | "
            f"podrazred: {podrazred} | "
            f"confidence: {confidence * 100:.2f}%"
        )

        logging.info(message)

        # Posodobi GUI, če teče (varno prek vrste).
        gui_queue.put(("result", result))
        gui_queue.put(("status", f"Zadnja napoved: {message}"))

        return message + "\n"

    except Exception as e:
        logging.exception("Prediction failed")
        gui_queue.put(("status", f"Napaka pri napovedi: {type(e).__name__}: {e}"))
        return f"FAIL: Prediction failed: {type(e).__name__}: {e}\n"
    

# =========================
# GUI
# =========================

class PredictionGUI:
    """
    Enak prikaz kot v gui.py (razred, podrazred, prepricanost, spektrogram),
    le da datoteke ni mogoče izbrati - signal pride iz STM32 prek servisa.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EkoAI - klasifikacija odpadkov (STM32 servis)")
        self.root.geometry("960x840")

        # Zgodovina prejetih vzorcev (vsak element je rezultat napovedi).
        self.samples = []
        self.current_index = -1

        # Pot do trenutne začasne WAV datoteke za predvajanje.
        self._wav_path = None

        self.status_var = tk.StringVar(value="Servis se zaganja...")
        self.counter_var = tk.StringVar(value="Vzorec: 0 / 0")
        self.razred_var = tk.StringVar(value="Razred: -")
        self.podrazred_var = tk.StringVar(value="Podrazred: -")
        self.prepricanost_var = tk.StringVar(value="Prepricanost: -")
        self.naprava_var = tk.StringVar(
            value=f"Naprava: {predictor.device}  |  Model: model9.pth"
        )

        self._zgradi_layout()
        self._osvezi_gumbe()

        # Ob zaprtju okna ustavi servis in predvajanje.
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Periodično preverjanje vrste z rezultati iz delovnih niti.
        self.root.after(100, self._poll_queue)

    def _zgradi_layout(self):
        zgornji = ttk.Frame(self.root, padding=10)
        zgornji.pack(fill="x")
        ttk.Label(zgornji, textvariable=self.status_var,
                  font=("Segoe UI", 11), wraplength=900).pack(side="left")

        # Kontrolna vrstica z gumbi.
        kontrole = ttk.Frame(self.root, padding=(10, 0, 10, 0))
        kontrole.pack(fill="x")

        self.next_btn = ttk.Button(kontrole, text="Naslednji vzorec",
                                   command=self.naslednji_vzorec)
        self.next_btn.pack(side="left")

        self.play_btn = ttk.Button(kontrole, text="Predvajaj",
                                   command=self.predvajaj)
        self.play_btn.pack(side="left", padx=(10, 0))

        self.stop_btn = ttk.Button(kontrole, text="Ustavi",
                                   command=self.ustavi)
        self.stop_btn.pack(side="left", padx=(5, 0))

        ttk.Label(kontrole, textvariable=self.counter_var,
                  font=("Segoe UI", 11)).pack(side="left", padx=15)

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
        self.ax.set_title("Spektrogram (cakam na snemanje)")
        self.ax.set_xlabel("Casovni okvir")
        self.ax.set_ylabel("Frekvencni bin")
        self.canvas.draw()

        spodnji = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        spodnji.pack(fill="x")
        ttk.Label(spodnji, textvariable=self.naprava_var,
                  foreground="#666").pack(anchor="w")

    def _poll_queue(self):
        nov_vzorec = False

        try:
            while True:
                kind, payload = gui_queue.get_nowait()

                if kind == "status":
                    self.status_var.set(payload)
                elif kind == "result":
                    # Nov vzorec samo shranimo v zgodovino - NE preskočimo
                    # samodejno naprej.
                    self.samples.append(payload)
                    nov_vzorec = True

        except queue.Empty:
            pass

        if nov_vzorec:
            # Prvi vzorec prikažemo samodejno, da okno ni prazno.
            if self.current_index == -1:
                self.current_index = 0
                self._prikazi_trenutni()

            self._osvezi_gumbe()

        self.root.after(100, self._poll_queue)

    # --- Navigacija ---

    def naslednji_vzorec(self):
        # Naprej se premaknemo le, če je na voljo novejši (nov) vzorec.
        if self.current_index < len(self.samples) - 1:
            self.current_index += 1
            self._prikazi_trenutni()
            self._osvezi_gumbe()
        else:
            self.status_var.set(
                "Ni novega vzorca - cakam na naslednje snemanje s STM32..."
            )

    def _prikazi_trenutni(self):
        if 0 <= self.current_index < len(self.samples):
            self._prikazi_rezultat(self.samples[self.current_index])

    def _osvezi_gumbe(self):
        skupaj = len(self.samples)
        prikazan = self.current_index + 1 if self.current_index >= 0 else 0
        cakajoci = skupaj - prikazan

        if cakajoci > 0:
            self.counter_var.set(f"Vzorec: {prikazan} / {skupaj}   (novih: {cakajoci})")
        else:
            self.counter_var.set(f"Vzorec: {prikazan} / {skupaj}")

        # "Naslednji vzorec" je aktiven le, če obstaja novejši vzorec.
        if self.current_index < skupaj - 1:
            self.next_btn.config(state="normal")
        else:
            self.next_btn.config(state="disabled")

        ima_vzorec = 0 <= self.current_index < skupaj
        play_stanje = "normal" if (ima_vzorec and winsound is not None) else "disabled"
        self.play_btn.config(state=play_stanje)
        self.stop_btn.config(state="normal" if winsound is not None else "disabled")

    # --- Predvajanje zvoka ---

    def predvajaj(self):
        if winsound is None:
            self.status_var.set("Predvajanje ni na voljo (winsound manjka).")
            return

        if not (0 <= self.current_index < len(self.samples)):
            return

        result = self.samples[self.current_index]
        signal = result.get("signal")
        Fvz = result.get("Fvz")

        if signal is None or Fvz is None:
            self.status_var.set("Vzorec nima zvocnih podatkov.")
            return

        try:
            # Ustavi morebitno prejšnje predvajanje in počisti staro datoteko.
            winsound.PlaySound(None, winsound.SND_PURGE)
            self._pocisti_wav()

            self._wav_path = self._zapisi_wav(signal, Fvz)
            winsound.PlaySound(
                self._wav_path,
                winsound.SND_FILENAME | winsound.SND_ASYNC
            )
            self.status_var.set(f"Predvajam vzorec {self.current_index + 1} ...")

        except Exception as e:
            self.status_var.set(f"Napaka pri predvajanju: {type(e).__name__}: {e}")

    def ustavi(self):
        if winsound is not None:
            winsound.PlaySound(None, winsound.SND_PURGE)
        self.status_var.set("Predvajanje ustavljeno.")

    def _zapisi_wav(self, signal, Fvz) -> str:
        data = np.asarray(signal, dtype=np.int16)

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="ekoai_")
        os.close(fd)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(int(round(Fvz)))
            wf.writeframes(data.tobytes())

        return path

    def _pocisti_wav(self):
        if self._wav_path:
            try:
                os.remove(self._wav_path)
            except OSError:
                pass
            self._wav_path = None

    def _on_close(self):
        if winsound is not None:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

        self._pocisti_wav()
        listen_stop_event.set()
        self.root.destroy()

    def _prikazi_rezultat(self, result: dict):
        razred = result["razred"]
        podrazred = result["podrazred"]
        conf = result["confidence"]

        self.razred_var.set(f"Razred: {razred}")
        self.podrazred_var.set(f"Podrazred: {podrazred}")
        self.prepricanost_var.set(f"Prepricanost: {conf * 100:.2f}%")

        naslov = f"vzorec {self.current_index + 1}" if self.current_index >= 0 else ""
        self._prikazi_spektrogram(result["spectrogram"], naslov)

    def _prikazi_spektrogram(self, spec, naslov: str = ""):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        im = self.ax.imshow(spec, aspect="auto", origin="lower", cmap="inferno")
        self.ax.set_title(f"Spektrogram - {naslov}" if naslov else "Spektrogram")
        self.ax.set_xlabel("Casovni okvir")
        self.ax.set_ylabel("Frekvencni bin")
        self.fig.colorbar(im, ax=self.ax)
        self.fig.tight_layout()
        self.canvas.draw()


# =========================
# TCP OBDELAVA UKAZOV
# =========================

def process_command(command):
    """
    Obdela en TCP ukaz.
    """

    try:
        command = command.strip()

        if command == "":
            return ""

        if command == "STATUS":
            if is_stm32_connected():
                return "STM32 is connected\n"
            else:
                return "STM32 is not connected\n"

        elif command == "GET_ALL":
            return get_all_logs()

        elif command == "GET_LAST":
            return get_last_log()

        elif command.startswith("GET_FILE|"):
            return get_file_by_name(command)

        elif command == "DELETE":
            return delete_files()

        elif command == "LISTEN":
            return start_listen()

        elif command == "STOP_LISTEN":
            return stop_listen()

        else:
            return f"FAIL: Unknown command: {command}\n"

    except Exception as e:
        logging.exception("Internal service error")
        return f"FAIL: Internal service error: {e}\n"


def handle_client(conn, addr):
    """
    Obdelava TCP klienta.
    """

    logging.info(f"Client connected: {addr}")

    try:
        with conn:
            stm32_port = find_stm32_port()

            if stm32_port is not None:
                conn.sendall(
                    f"Connected to SPO STM32 service - STM32 detected at {stm32_port}\n".encode()
                )
            else:
                conn.sendall(
                    b"Connected to SPO STM32 service - No STM32 detected\n"
                )

            buffer = ""

            while True:
                try:
                    data = conn.recv(1024)

                    if not data:
                        break

                    decoded = data.decode(errors="replace")

                    # Če klient slučajno pošlje Ctrl+C kot ETX znak.
                    # V nekaterih terminalih se to ne zgodi, ker Ctrl+C zapre klienta.
                    if "\x03" in decoded:
                        response = stop_listen()
                        conn.sendall(response.encode())
                        decoded = decoded.replace("\x03", "")

                    buffer += decoded

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        command = line.strip()

                        if command == "":
                            continue

                        logging.info(f"TCP command from {addr}: {command}")

                        response = process_command(command)

                        if response:
                            conn.sendall(response.encode())

                except ConnectionResetError:
                    break

                except OSError as e:
                    logging.error(f"Client socket error: {e}")
                    break

                except Exception as e:
                    logging.exception("Unexpected client handling error")

                    try:
                        conn.sendall(f"FAIL: Client handling error: {e}\n".encode())
                    except Exception:
                        pass

                    break

    finally:
        logging.info(f"Client disconnected: {addr}")


# =========================
# MAIN SERVER
# =========================

def run_server():
    """
    TCP strežnik. Teče v ozadju, da GUI lahko zaseda glavno nit.
    """

    logging.info(f"Service starting on {HOST}:{SERVICE_PORT}")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((HOST, SERVICE_PORT))
            server.listen()

            logging.info(f"Service listening on {HOST}:{SERVICE_PORT}")
            gui_queue.put((
                "status",
                f"Servis posluša na {HOST}:{SERVICE_PORT}. Cakam na snemanje s STM32..."
            ))

            while True:
                try:
                    conn, addr = server.accept()

                    thread = threading.Thread(
                        target=handle_client,
                        args=(conn, addr),
                        daemon=True
                    )

                    thread.start()

                except OSError as e:
                    logging.error(f"Server socket error: {e}")
                    time.sleep(1)

                except Exception as e:
                    logging.exception(f"Unexpected server error: {e}")
                    time.sleep(1)

    except Exception as e:
        logging.exception(f"Server stopped with error: {e}")

    finally:
        listen_stop_event.set()
        logging.info("Service stopped")


def main():
    # TCP strežnik teče v ozadju.
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # GUI teče na glavni niti.
    root = tk.Tk()
    PredictionGUI(root)

    try:
        root.mainloop()
    finally:
        listen_stop_event.set()
        logging.info("GUI closed, service stopping")


if __name__ == "__main__":
    main()
