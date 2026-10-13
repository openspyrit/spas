# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 2026

@author: mahieu

Laser combiner (6 lasers), driven through the creatis_pilot_combiner package.
"""

import os
import sys
import json
import time
import tempfile
import threading
import subprocess
from multiprocessing.connection import Listener, Client
import tkinter as tk
from tkinter import messagebox
from dataclasses import dataclass, field
from typing import List, Optional
from dataclasses_json import dataclass_json

COMBINER_LIB_PATH = r"D:\materiel\gitLab\toolbox\Combiner\python_lib\src"
if COMBINER_LIB_PATH not in sys.path:
    sys.path.append(COMBINER_LIB_PATH)

try:
    from creatis_pilot_combiner import Combiner as CombinerDriver
except ImportError:   # package only needed to drive the lasers, not to read the metadata
    CombinerDriver = None

MIN_POWER_MW = 1   # minimum power accepted by the driver (0 is accepted too)


@dataclass_json
@dataclass
class Combiner_Parameters:
    """
    Class containing the laser combiner parameters, read from the lasers.
    lasers : one dict per connected laser, sorted by increasing wavelength :
        {'wavelength_nm', 'max_power_mW', 'power_mW', 'power_percent', 'state' ('ON' or 'OFF')}
    """
    class_description: str = 'laser combiner parameters'
    model: Optional[str] = None
    serial_number: Optional[str] = None
    lasers: List[dict] = field(default_factory = list)


class Combiner:
    """Class that controls the laser combiner. The lasers are detected automatically on the serial ports
    and called by their wavelength (nm): the closest connected wavelength within WAVELENGTH_TOLERANCE_NM is used.

    Example:
        combi = Combiner()
        combi.init(model = 'combiner', SN = '2')
        combi.laser_on(405)
        combi.set_power(405, 10, unit = '%')
        combi.laser_off(405)
        combi.control()                      # window to control the lasers
        combi_params = combi.get_params()   # metadata : power and ON/OFF state of each laser
        combi.disconnect()
    """

    WAVELENGTH_TOLERANCE_NM = 5

    def __init__(self):

        self.device = None
        self.model = None
        self.serial_number = None
        self.lasers = {}   # {wavelength (nm): {'max_power': (mW), 'idn': combiner idn}}, sorted by increasing wavelength
        self.lock = threading.RLock()   # the control window and the console share the serial ports
        self._gui = None                # process of the control window (subprocess.Popen)
        self._gui_conn = None           # connection with the control window


    def init(self, model: str = 'combiner', SN: str = '2', default_power: float = 5, verbose: bool = True):
        """
        Initialize the laser combiner and detect the connected lasers

        Parameters
        ----------
        model : str
            the model of the combiner. The default is 'combiner'.
        SN : str
            the identifier of the combiner. The default is '2'.
        default_power : float
            power (%) set to all the lasers at the initialization. The default is 5 %. None : not changed.
        verbose : bool
            print the connected lasers. The default is True.
            The detected lasers are stored in self.lasers.
        """
        if CombinerDriver is None:
            raise ImportError('creatis_pilot_combiner package not found in ' + COMBINER_LIB_PATH)
        self.device = CombinerDriver()
        self.device.init_lasers()
        self.model = model
        self.serial_number = str(SN)

        if self.device.n_laser == 0:
            print('Warning, combiner ' + model + ', id = ' + self.serial_number + ' : no laser detected')

        all_lasers = [laser for key in self.device.laser for laser in self.device.laser[key]]
        all_lasers.sort(key = lambda laser: laser.wavelength)
        self.lasers = {}
        for laser in all_lasers:
            self.lasers[laser.wavelength] = {'max_power': laser.max_power, 'idn': laser.idn}

        print('Combiner ' + model + ', id = ' + self.serial_number + ' connected')
        if verbose:
            self.print_lasers()

        self.all_off()

        if default_power is not None:
            for wavelength in self.lasers:
                try:
                    self.set_power(wavelength, *self.clamp_power(wavelength, default_power, '%'))
                except Exception as e:
                    print('Warning, laser ' + f"{wavelength:g}" + ' nm : default power not set (' + str(e) + ')')

    def all_off(self):
        """Switch OFF all the lasers. Obis, Zouk : state read, OFF sent only if ON (each command is written
        in the memory of the Obis). LuxX : state not readable, considered ON and OFF always sent.
        A message is printed only if a laser was detected ON or could not be switched OFF."""
        all_ok = True
        found_on = False
        for wavelength in self.lasers:
            laser = self._laser(wavelength)
            try:
                with self.lock:
                    if hasattr(laser, 'get_emission_state'):   # Obis, Zouk
                        if not laser.get_emission_state():
                            continue
                        found_on = True
                        print('Warning, laser ' + f"{wavelength:g}" + ' nm was ON')
                    else:                                      # LuxX
                        laser.output = True   # state unknown : considered ON, so that OFF is really sent
                self.laser_off(wavelength)
            except Exception as e:
                all_ok = False
                print('Warning, laser ' + f"{wavelength:g}" + ' nm : not switched OFF (' + str(e) + ')')
        if found_on and all_ok:
            print('All the lasers are OFF')

    def print_lasers(self):

        for nbr, (wavelength, laser) in enumerate(self.lasers.items(), start = 1):
            max_power = '?' if laser['max_power'] is None else f"{laser['max_power']:g}"
            print('Laser ' + str(nbr) + ': (' + f"{wavelength:g}" + ' nm, ' + max_power + ' mW) connected')

    def _idn(self, wavelength: float):
        """Return the combiner idn of the connected laser closest to wavelength (nm)."""
        available = list(self.lasers.keys())
        if available:
            closest = min(available, key = lambda w: abs(w - wavelength))
            if abs(closest - wavelength) <= self.WAVELENGTH_TOLERANCE_NM:
                return self.lasers[closest]['idn']
        raise ValueError('Laser ' + f"{wavelength:g}" + ' nm not connected, available wavelengths (nm) : ' +
                         str([f"{w:g}" for w in available]))

    def _laser(self, wavelength: float):
        """Return the laser object of the driver for the given wavelength (nm)."""
        key, index = self.device.get_laser_key_and_index(idn = self._idn(wavelength))
        return self.device.laser[key][index]

    def _state(self, laser):
        """Read the output state of a laser : TTL line if configured, else the laser itself."""
        ttl_port = getattr(laser, 'ttl_port', None)
        if ttl_port is not None and self.device.ni_board is not None:
            return self.device.ni_board.get_state(ttl_port)
        if hasattr(laser, 'get_emission_state'):   # Obis, Zouk
            return laser.get_emission_state()
        return laser.output                        # LuxX : last state sent

    def get_params(self, verbose: bool = False):
        """
        Read the power and the ON/OFF state of all the connected lasers (to be called just before
        the acquisition, as the powers and states change during the program)

        Parameters
        ----------
        verbose : bool
            print the parameters. The default is False.

        Returns
        -------
        combi_params : Combiner_Parameters
        """
        combi_params = Combiner_Parameters(model = self.model, serial_number = self.serial_number)
        for wavelength, laser_info in self.lasers.items():
            laser = self._laser(wavelength)
            power_percent, state = self.read_laser(wavelength)
            state = '?' if state is None else ('ON' if state else 'OFF')
            combi_params.lasers.append({'wavelength_nm': wavelength,
                                        'max_power_mW': laser_info['max_power'],
                                        'power_mW': getattr(laser, 'power_mw', None),
                                        'power_percent': getattr(laser, 'power_pourcent', None),
                                        'state': state})
        if verbose:
            for laser in combi_params.lasers:
                power_mW = '?' if laser['power_mW'] is None else f"{laser['power_mW']:.4g}"
                power_percent = '?' if laser['power_percent'] is None else f"{laser['power_percent']:.3g}"
                print('Laser ' + f"{laser['wavelength_nm']:g}" + ' nm : ' + laser['state'] + ', ' + power_mW + ' mW (' + power_percent + ' %)')

        return combi_params

    def read_laser(self, wavelength: float):
        """
        Read the power (%) and the output state of a laser

        Returns
        -------
        power_percent : float (last value kept if not read)
        state : bool (True = ON), None if not read
        """
        laser = self._laser(wavelength)
        with self.lock:
            try:
                laser.get_power_level()
            except Exception as e:
                print('Warning, power of the laser ' + f"{wavelength:g}" + ' nm not read (' + str(e) + '), last value kept')
            try:
                state = bool(self._state(laser))
            except Exception as e:
                print('Warning, state of the laser ' + f"{wavelength:g}" + ' nm not read (' + str(e) + ')')
                state = None
        return getattr(laser, 'power_pourcent', None), state

    def laser_on(self, wavelength: float):

        with self.lock:
            self.device.laser_on(idn = self._idn(wavelength))

    def laser_off(self, wavelength: float):

        with self.lock:
            self.device.laser_off(idn = self._idn(wavelength))

    def set_power(self, wavelength: float, power: float, unit: str = '%'):
        """
        Set the power of a laser

        Parameters
        ----------
        wavelength : float
            the wavelength of the laser (nm), e.g. 405.
        power : float
            the power.
        unit : str
            '%' or 'mW'. The default is '%'.
        """
        laser = self._laser(wavelength)
        if power == 0 and getattr(laser, 'min_power', None):   # Obis : 0 raised by the laser to its minimum
            print(f"Warning, laser {wavelength:g} nm : 0 is raised by the laser to its minimum power ({laser.min_power} mW)")
        with self.lock:
            self.device.set_laser_power(power, unit = unit, idn = self._idn(wavelength))

    def clamp_power(self, wavelength: float, power: float, unit: str = '%', verbose: bool = True):
        """Return (power, unit) raised to the minimum accepted by the driver (MIN_POWER_MW) if the
        requested power is strictly between 0 and this minimum. 0 is accepted."""
        max_power = self.lasers[min(self.lasers, key = lambda w: abs(w - wavelength))]['max_power']
        power_mW = power if unit == 'mW' else (None if max_power is None else power / 100 * max_power)
        if power_mW is not None and 0 < power_mW < MIN_POWER_MW:
            if verbose:
                print(f"Laser {wavelength:g} nm : power set to the minimum, {MIN_POWER_MW} mW")
            return MIN_POWER_MW, 'mW'
        return power, unit

    def control(self, x: Optional[int] = None, y: int = 0):
        """Open the window to control the lasers. For each laser : ON/OFF button, power gauge (left : %,
        right : mW) and two boxes to enter the power by hand, one in % and one in mW (validated with Enter).

        The window runs in its own Python process (tkinter in a thread of the kernel crashes it, with the
        threads of the other instruments) : the console stays available. This process only imports this
        module (subprocess, not multiprocessing which would run the main script again). The serial ports
        stay in the kernel, a thread (without tkinter) sends the commands of the window to the lasers and
        their state to the window, through a local connection.

        Parameters
        ----------
        x, y : int, optional
            the position (pixels) of the top left corner of the window on the screen.
            The default is None : position chosen by Windows.
        """
        if not self.lasers:
            print('No laser connected, run combi.init() first')
            return
        if self._gui is not None and self._gui.poll() is None:
            print('The control window is already open')
            return

        key = os.urandom(16)
        listener = Listener(('127.0.0.1', 0), authkey = key)   # local connection with the window
        settings = {'port': listener.address[1], 'title': str(self.model) + ', id = ' + str(self.serial_number),
                    'lasers': [[wavelength, info['max_power']] for wavelength, info in self.lasers.items()], 'x': x, 'y': y}
        env = dict(os.environ, COMBINER_WINDOW_KEY = key.hex())
        log = tempfile.NamedTemporaryFile('w+', prefix = 'combiner_window_', suffix = '.log', delete = False)
        self._gui = subprocess.Popen([sys.executable, '-c', 'from spas.combiner_module import _control_window_main; _control_window_main()',
                                      json.dumps(settings)], env = env, stdout = log, stderr = subprocess.STDOUT,
                                     creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        threading.Thread(target = self._serve_window, args = (listener, log), daemon = True).start()

    def _connect_window(self, listener, log):
        """Wait for the connection of the window (20 s max). Return the connection, None if it failed."""
        process = self._gui
        connected = threading.Event()

        def watchdog():   # accept() has no timeout : the listener is closed if the window does not connect
            t0 = time.time()
            while not connected.is_set() and process.poll() is None and time.time() - t0 < 20:
                time.sleep(0.1)
            if not connected.is_set():
                listener.close()

        threading.Thread(target = watchdog, daemon = True).start()
        try:
            conn = listener.accept()
            connected.set()
            return conn
        except Exception:
            time.sleep(0.5)
            log.seek(0)
            print('Control window of the lasers : not opened.' + nl_text(log.read()))
            if process.poll() is None:
                process.kill()
            return None
        finally:
            listener.close()

    def _laser_state(self, wavelength: float):
        """Last known power (%) and state of a laser, without serial communication (state None if not known)."""
        laser = self._laser(wavelength)
        state = None if getattr(laser, 'ttl_port', None) is not None else getattr(laser, 'output', None)
        return getattr(laser, 'power_pourcent', None), state

    def _serve_window(self, listener, log):
        """Thread of the kernel (no tkinter) : executes the commands of the control window and sends it
        the state of the lasers (read after each command, last known values every second)."""
        conn = self._connect_window(listener, log)
        if conn is None:
            return
        self._gui_conn = conn
        try:
            for wavelength in self.lasers:
                conn.send(('laser', wavelength) + tuple(self.read_laser(wavelength)))
            last_update = time.time()
            while self.device is not None:
                if conn.poll(0.1):
                    message = conn.recv()
                    if message[0] == 'closed':
                        break
                    command, wavelength = message[0], message[1]
                    try:
                        if command == 'power':
                            self.set_power(wavelength, *self.clamp_power(wavelength, message[2], message[3]))
                        elif command == 'on':
                            self.laser_on(wavelength)
                        elif command == 'off':
                            self.laser_off(wavelength)
                    except Exception as e:
                        conn.send(('error', f"Laser {wavelength:g} nm : {e}"))
                    conn.send(('laser', wavelength) + tuple(self.read_laser(wavelength)))
                if time.time() - last_update > 1 and self.device is not None:
                    conn.send(('state', {wavelength: self._laser_state(wavelength) for wavelength in self.lasers}))
                    last_update = time.time()
        except (EOFError, OSError):   # window closed or crashed
            pass
        except Exception as e:
            print('Control window of the lasers : communication stopped (' + str(e) + ')')
        finally:
            conn.close()
            self._gui_conn = None

    def _close_window(self):
        """Close the control window, if it is open."""
        if self._gui is not None and self._gui.poll() is None:
            try:
                self._gui_conn.send(('close',))
            except Exception:
                pass
            try:
                self._gui.wait(2)
            except subprocess.TimeoutExpired:
                self._gui.kill()
        self._gui = None

    def disconnect(self):
        """Close the control window, set all the lasers to 0 W, then OFF, and close."""
        self._close_window()
        for wavelength in self.lasers:
            try:
                self.set_power(wavelength, 0, 'mW')
            except Exception as e:
                print('Warning, laser ' + f"{wavelength:g}" + ' nm : power not set to 0 (' + str(e) + ')')
            try:
                self.laser_off(wavelength)
            except Exception as e:
                print('Warning, laser ' + f"{wavelength:g}" + ' nm : not switched OFF (' + str(e) + ')')
        print('Lasers set to 0 W and OFF')

        with self.lock:
            self.device.close()
            self.device = None
        self.lasers = {}
        print('Combiner disconnected')


def wavelength_to_color(wavelength: float) -> str:
    """Approximate color (hex) of a wavelength (nm), for the display. Gray outside the visible range."""
    w = wavelength
    if 380 <= w < 440:
        r, g, b = (440 - w) / 60, 0, 1
    elif 440 <= w < 490:
        r, g, b = 0, (w - 440) / 50, 1
    elif 490 <= w < 510:
        r, g, b = 0, 1, (510 - w) / 20
    elif 510 <= w < 580:
        r, g, b = (w - 510) / 70, 1, 0
    elif 580 <= w < 645:
        r, g, b = 1, (645 - w) / 65, 0
    elif 645 <= w <= 780:
        r, g, b = 1, 0, 0
    else:
        return '#808080'
    return '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))


def text_color(background: str) -> str:
    """Black or white, the most readable on the background color (hex)."""
    r, g, b = (int(background[i:i + 2], 16) for i in (1, 3, 5))
    return 'black' if 0.299 * r + 0.587 * g + 0.114 * b > 140 else 'white'


def nl_text(text: str) -> str:
    """Text preceded by a new line, '' if empty (for the error messages)."""
    text = text.strip()
    return '\n' + text if text else ''


def _control_window_main():
    """Entry point of the process of the control window (tkinter in its main thread), launched by
    Combiner.control() : settings in argv (json), key of the connection in the environment."""
    settings = json.loads(sys.argv[1])
    conn = Client(('127.0.0.1', settings['port']), authkey = bytes.fromhex(os.environ['COMBINER_WINDOW_KEY']))
    lasers = {wavelength: max_power for wavelength, max_power in settings['lasers']}
    CombinerControl(conn, lasers, settings['title'], settings['x'], settings['y'] or 0).run()


class CombinerControl:
    """Window to control the lasers of the combiner: ON/OFF, power gauge (left : %, right : mW), manual values.
    It runs in its own process and communicates with the kernel (Combiner._serve_window) through conn :
    it sends ('power', wavelength, value, unit), ('on', wavelength), ('off', wavelength), ('closed',) and
    receives ('laser', wavelength, percent, state), ('state', {wavelength: (percent, state)}), ('error', text), ('close',)."""

    GAUGE_H = 200          # height of the gauge (pixels)
    GAUGE_MARGIN = 10      # margin above and below the gauge
    TROUGH_X = (42, 58)    # left and right of the trough
    CANVAS_W = 110
    TICKS = (0, 25, 50, 75, 100)   # (%)

    def __init__(self, conn, lasers: dict, title: str, x: Optional[int] = None, y: int = 0):
        self.conn = conn
        self.widgets = {}      # {wavelength: widgets and displayed values of the laser}
        self.pending = set()   # lasers waiting for the answer to a command (the periodic states are ignored)

        self.root = tk.Tk()
        self.root.title(title)
        self.root.attributes('-topmost', True)
        self.root.resizable(False, False)
        if x is not None:
            self.root.geometry('+' + str(int(x)) + '+' + str(int(y)))
        self.root.protocol('WM_DELETE_WINDOW', self.close)

        frame = tk.Frame(self.root, padx = 5, pady = 5)
        frame.pack()
        for col, (wavelength, max_power) in enumerate(lasers.items()):
            self._build_laser(frame, col, wavelength, max_power)

        tk.Button(self.root, text = 'CLOSE', command = self.close).pack(pady = (0, 8), fill = 'x', padx = 8)

        self.root.after(100, self._poll)

    # ----- construction -----
    def _build_laser(self, parent, col, wavelength, max_power):
        color = wavelength_to_color(wavelength)
        box = tk.LabelFrame(parent, text = f"{wavelength:g} nm", font = ('Arial', 10, 'bold'), padx = 3, pady = 3)
        box.grid(row = 0, column = col, padx = 3)

        # two boxes to enter the power by hand (Enter to validate) : in % and in mW
        entries = tk.Frame(box)
        entries.pack()
        entry = {}
        for column, unit in enumerate(('%', 'mW')):
            entry[unit] = tk.Entry(entries, width = 5, justify = 'center')
            entry[unit].grid(row = 0, column = 2 * column)
            entry[unit].bind('<Return>', lambda event, w = wavelength, u = unit: self._entry_apply(w, u))
            tk.Label(entries, text = unit, font = ('Arial', 8)).grid(row = 0, column = 2 * column + 1, padx = (1, 3))

        canvas = tk.Canvas(box, width = self.CANVAS_W, height = self.GAUGE_H + 2 * self.GAUGE_MARGIN, highlightthickness = 0)
        canvas.pack()
        self._draw_scale(canvas, max_power)
        fill = canvas.create_rectangle(self.TROUGH_X[0] + 1, self._y(0), self.TROUGH_X[1] - 1, self._y(0), fill = color, width = 0)
        cursor = canvas.create_rectangle(self.TROUGH_X[0] - 6, self._y(0) - 3, self.TROUGH_X[1] + 6, self._y(0) + 3,
                                         fill = '#404040', outline = 'white')
        canvas.bind('<Button-1>', lambda event, w = wavelength: self._drag(w, event))
        canvas.bind('<B1-Motion>', lambda event, w = wavelength: self._drag(w, event))
        canvas.bind('<ButtonRelease-1>', lambda event, w = wavelength: self._drag_apply(w))

        button = tk.Button(box, text = 'OFF', width = 8, font = ('Arial', 10, 'bold'), command = lambda w = wavelength: self._toggle(w))
        button.pack(pady = (3, 0))

        self.widgets[wavelength] = {'entry': entry, 'canvas': canvas, 'fill': fill, 'cursor': cursor, 'button': button,
                                    'max_power': max_power, 'percent': 0.0, 'state': False, 'dragging': False}

    def _y(self, percent):
        return self.GAUGE_MARGIN + self.GAUGE_H * (1 - percent / 100)

    def _draw_scale(self, canvas, max_power):
        x0, x1 = self.TROUGH_X
        canvas.create_rectangle(x0, self._y(100), x1, self._y(0), fill = '#e8e8e8', outline = 'gray')
        for percent in self.TICKS:
            y = self._y(percent)
            canvas.create_line(x0 - 4, y, x0, y)
            canvas.create_line(x1, y, x1 + 4, y)
            canvas.create_text(x0 - 6, y, text = f"{percent} %", anchor = 'e', font = ('Arial', 7))
            power = '?' if max_power is None else f"{percent / 100 * max_power:.3g}"
            canvas.create_text(x1 + 6, y, text = power + ' mW', anchor = 'w', font = ('Arial', 7))

    # ----- display -----
    def _show(self, wavelength, percent = None, state = None):
        wid = self.widgets[wavelength]
        if percent is not None:
            wid['percent'] = min(max(percent, 0), 100)
            y = self._y(wid['percent'])
            wid['canvas'].coords(wid['fill'], self.TROUGH_X[0] + 1, y, self.TROUGH_X[1] - 1, self._y(0))
            wid['canvas'].coords(wid['cursor'], self.TROUGH_X[0] - 6, y - 3, self.TROUGH_X[1] + 6, y + 3)
            power_mW = None if wid['max_power'] is None else wid['percent'] / 100 * wid['max_power']
            values = {'%': f"{wid['percent']:.3g}", 'mW': '?' if power_mW is None else f"{power_mW:.3g}"}
            focus = self.root.focus_get()
            for unit, entry in wid['entry'].items():
                if focus is not entry:   # do not erase a value being typed
                    entry.delete(0, 'end')
                    entry.insert(0, values[unit])
        if state is not None:
            wid['state'] = state
            color = wavelength_to_color(wavelength)
            wid['button'].config(text = 'ON' if state else 'OFF', bg = color if state else '#d0d0d0',
                                 fg = text_color(color) if state else 'black', activebackground = color if state else '#e0e0e0',
                                 activeforeground = text_color(color) if state else 'black')

    def _poll(self):
        """Messages from the kernel."""
        try:
            while self.conn.poll():
                message = self.conn.recv()
                if message[0] == 'laser':   # answer to a command : values read on the laser
                    wavelength = message[1]
                    self.pending.discard(wavelength)
                    if not self.widgets[wavelength]['dragging']:
                        self._show(wavelength, message[2], message[3])
                elif message[0] == 'state':   # last known values, in case they are changed from the console
                    for wavelength, (percent, state) in message[1].items():
                        if wavelength not in self.pending and not self.widgets[wavelength]['dragging']:
                            self._show(wavelength, percent, state)
                elif message[0] == 'error':
                    messagebox.showerror('Error', message[1], parent = self.root)
                elif message[0] == 'close':
                    self.root.destroy()
                    return
        except (EOFError, OSError):   # kernel stopped
            self.root.destroy()
            return
        self.root.after(100, self._poll)

    # ----- actions -----
    def _send(self, *message):
        self.pending.add(message[1])
        try:
            self.conn.send(message)
        except (EOFError, OSError):   # kernel stopped
            self.root.destroy()

    def _drag(self, wavelength, event):
        self.widgets[wavelength]['dragging'] = True
        percent = (self.GAUGE_MARGIN + self.GAUGE_H - event.y) / self.GAUGE_H * 100
        self._show(wavelength, percent)

    def _drag_apply(self, wavelength):
        wid = self.widgets[wavelength]
        wid['dragging'] = False
        self._send('power', wavelength, round(wid['percent'], 1), '%')

    def _entry_apply(self, wavelength, unit):
        wid = self.widgets[wavelength]
        try:
            value = float(wid['entry'][unit].get().strip().replace(',', '.'))
        except ValueError:
            messagebox.showerror('Error', f"Laser {wavelength:g} nm : enter a number (in {unit})", parent = self.root)
            return
        max_value = 100 if unit == '%' else wid['max_power']
        if unit == 'mW' and max_value is None:
            messagebox.showerror('Error', f"Laser {wavelength:g} nm : max power unknown, enter the power in %", parent = self.root)
            return
        if not 0 <= value <= max_value:
            messagebox.showerror('Error', f"Laser {wavelength:g} nm : value out of range (0 - {max_value:g} {unit})", parent = self.root)
            return
        self.root.focus_set()   # leave the box : it is updated with the value read on the laser
        self._send('power', wavelength, value, unit)

    def _toggle(self, wavelength):
        self._send('off' if self.widgets[wavelength]['state'] else 'on', wavelength)

    def close(self):
        try:
            self.conn.send(('closed',))
        except (EOFError, OSError):
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()
