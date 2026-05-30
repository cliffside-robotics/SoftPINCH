import socket
import threading
import csv
import time
import math
from queue import Queue, Empty


COUNTS_PER_REV = 1494.0
COUNTS_TO_RAD = (2 * math.pi) / COUNTS_PER_REV


class ESPLogger:
    def __init__(self, host="10.126.128.231", port=1234):
        self.host = host
        self.port = port

        self.running = True
        self.buffer = ""

        self.csvfile = None
        self.writer = None

        self.csv_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.events = Queue()

        self.session_active = False
        self.motion_state = "released"
        # possible values:
        # "released", "contract_requested", "contracting",
        # "contracted", "release_requested", "releasing"

        self.current_cycle = 0
        self.pending_command = None

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))
        self.s.sendall(b"connect\n")

        self.receiver_thread = threading.Thread(
            target=self._receiver,
            daemon=True
        )
        self.receiver_thread.start()

    # -------------------------
    # Public API
    # -------------------------

    def start_session(self, filename=None):
        """
        Starts a session and creates the CSV file.
        Non-blocking: returns immediately after sending the command.
        """
        with self.csv_lock:
            if self.csvfile is not None:
                return False

            if filename is None:
                filename = time.strftime("session_%Y%m%d_%H%M%S.csv")

            self.csvfile = open(filename, "w", newline="")
            self.writer = csv.writer(self.csvfile)

            self.writer.writerow([
                "t_ms",
                "encoder_count",
                "position_rad",
                "ADC_Index",
                "ADC_Thumb",
                "Voltage_Index",
                "Voltage_Thumb",
                "X_gauss",
                "Y_gauss",
                "Z_gauss",
                "test"
            ])

        self._send_command("startsession")
        return True

    def contract(self):
        """
        Sends a contract command if locally allowed.
        Non-blocking: returns immediately.
        Returns True if the command was sent, False if ignored locally.
        """
        with self.state_lock:
            if (
                not self.session_active
                or self.motion_state != "released"
                or self.pending_command is not None
            ):
                return False

            self.pending_command = "contract"
            self.motion_state = "contract_requested"

        self._send_command("contract")
        return True

    def release(self):
        """
        Sends a release command if locally allowed.
        Non-blocking: returns immediately.
        Returns True if the command was sent, False if ignored locally.
        """
        with self.state_lock:
            if (
                not self.session_active
                or self.motion_state != "contracted"
                or self.pending_command is not None
            ):
                return False

            self.pending_command = "release"
            self.motion_state = "release_requested"

        self._send_command("release")
        return True

    def end_session(self):
        """
        Ends the session if the ESP32 is released and idle.
        Non-blocking: returns immediately.
        """
        with self.state_lock:
            if self.motion_state != "released":
                return False

        self._send_command("endsession")
        return True

    def pop_events(self):
        """
        Returns all pending events collected from the ESP32.
        """
        events = []

        while True:
            try:
                events.append(self.events.get_nowait())
            except Empty:
                break

        return events

    def close(self):
        """
        Closes socket and CSV safely.
        """
        self.running = False

        with self.csv_lock:
            if self.csvfile:
                self.csvfile.close()
                self.csvfile = None
                self.writer = None

        try:
            self.s.close()
        except OSError:
            pass

    @property
    def is_released(self):
        with self.state_lock:
            return self.motion_state == "released"

    @property
    def is_contracted(self):
        with self.state_lock:
            return self.motion_state == "contracted"

    @property
    def is_moving(self):
        with self.state_lock:
            return self.motion_state in {
                "contract_requested",
                "contracting",
                "release_requested",
                "releasing"
            }

    # -------------------------
    # Internal methods
    # -------------------------

    def _send_command(self, cmd):
        self.s.sendall((cmd + "\n").encode())

    def _receiver(self):
        while self.running:
            try:
                data = self.s.recv(4096).decode(errors="replace")

                if not data:
                    break

                self.buffer += data

                while "\n" in self.buffer:
                    line, self.buffer = self.buffer.split("\n", 1)
                    line = line.strip()

                    if not line:
                        continue

                    if line.startswith("DATA"):
                        self._handle_data_line(line)
                    else:
                        self._handle_status_line(line)

            except Exception as e:
                if self.running:
                    self.events.put({
                        "type": "receiver_error",
                        "message": str(e)
                    })
                break

    def _handle_data_line(self, line):
        p = line.split(",")

        if len(p) != 9:
            return

        try:
            _, t_ms, enc, adc_index, adc_thumb, scaledX, scaledY, scaledZ, test = p

            t_ms = int(t_ms)
            enc = int(enc)
            adc_index = int(adc_index)
            adc_thumb = int(adc_thumb)
            scaledX = float(scaledX)
            scaledY = float(scaledY)
            scaledZ = float(scaledZ)
            test = int(test)

        except ValueError:
            return

        pos = enc * COUNTS_TO_RAD

        volt_index = adc_index * 3.3 / 4095.0
        volt_thumb = adc_thumb * 3.3 / 4095.0

        X_gauss = scaledX * 8.0
        Y_gauss = scaledY * 8.0
        Z_gauss = scaledZ * 8.0

        with self.csv_lock:
            if self.writer:
                self.writer.writerow([
                    t_ms,
                    enc,
                    pos,
                    adc_index,
                    adc_thumb,
                    volt_index,
                    volt_thumb,
                    X_gauss,
                    Y_gauss,
                    Z_gauss,
                    test
                ])

    def _handle_status_line(self, line):
        print("ESP:", line)

        if line == "SESSION STARTED":
            with self.state_lock:
                self.session_active = True
                self.motion_state = "released"
                self.pending_command = None
                self.current_cycle = 0

            self.events.put({
                "type": "session_started"
            })

        elif line == "SESSION DONE":
            with self.state_lock:
                self.session_active = False
                self.pending_command = None

            with self.csv_lock:
                if self.csvfile:
                    self.csvfile.close()
                    self.csvfile = None
                    self.writer = None

            self.events.put({
                "type": "session_done"
            })

        elif line.startswith("MOTION_STARTED"):
            parts = line.split(",")

            if len(parts) == 3:
                _, motion, cycle = parts

                try:
                    cycle = int(cycle)
                except ValueError:
                    cycle = None

                with self.state_lock:
                    self.current_cycle = cycle
                    self.pending_command = None

                    if motion == "contract":
                        self.motion_state = "contracting"
                    elif motion == "release":
                        self.motion_state = "releasing"

                self.events.put({
                    "type": "motion_started",
                    "motion": motion,
                    "cycle": cycle
                })

        elif line.startswith("MOTION_DONE"):
            parts = line.split(",")

            if len(parts) == 3:
                _, motion, cycle = parts

                try:
                    cycle = int(cycle)
                except ValueError:
                    cycle = None

                with self.state_lock:
                    self.current_cycle = cycle
                    self.pending_command = None

                    if motion == "contract":
                        self.motion_state = "contracted"
                    elif motion == "release":
                        self.motion_state = "released"

                self.events.put({
                    "type": "motion_done",
                    "motion": motion,
                    "cycle": cycle
                })

        elif line.startswith("STOP_CAUSE:"):
            cause = line.split(":", 1)[1].strip()

            self.events.put({
                "type": "stop_cause",
                "cause": cause
            })

        elif line == "IGNORED: CONTRACT":
            with self.state_lock:
                self.pending_command = None

                if self.motion_state == "contract_requested":
                    self.motion_state = "released"

            self.events.put({
                "type": "ignored",
                "command": "contract"
            })

        elif line == "IGNORED: RELEASE":
            with self.state_lock:
                self.pending_command = None

                if self.motion_state == "release_requested":
                    self.motion_state = "contracted"

            self.events.put({
                "type": "ignored",
                "command": "release"
            })

        elif line == "IGNORED: ENDSESSION_NOT_RELEASED":
            self.events.put({
                "type": "ignored",
                "command": "endsession"
            })

        elif line == "IGNORED: STARTSESSION_NOT_RELEASED":
            self.events.put({
                "type": "ignored",
                "command": "startsession"
            })

        else:
            self.events.put({
                "type": "esp_message",
                "message": line
            })