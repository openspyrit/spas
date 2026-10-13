# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 2026

@author: mahieu

Laser combiner (6 lasers), driven through the creatis_pilot_combiner package.
"""

import sys
import threading
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

        if default_power is not None:
            for wavelength in self.lasers:
                try:
                    self.set_power(wavelength, *self.clamp_power(wavelength, default_power, '%'))
                except Exception as e:
                    print('Warning, laser ' + f"{wavelength:g}" + ' nm : default power not set (' + str(e) + ')')
            print('Power of the lasers set to ' + f"{default_power:g}" + ' %')

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
        """Open the window to control the lasers, in a thread: the console stays available.
        For each laser : ON/OFF button, power gauge (left : %, right : mW) and a box to enter the
        power by hand ('20' => 20 %, '20 mW' => 20 mW, validated with Enter).

        Parameters
        ----------
        x, y : int, optional
            the position (pixels) of the top left corner of the window on the screen.
            The default is None : position chosen by Windows.
        """
        if not self.lasers:
            print('No laser connected, run combi.init() first')
            return

        def start_gui():
            gui = CombinerControl(self, x, y)
            gui.run() # The mainloop runs here, in its own thread

        # daemon=True lets the window close when Python is closed
        threading.Thread(target = start_gui, daemon = True).start()

    def disconnect(self):
        """Set all the lasers to 0 W, then OFF, and close."""
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


class CombinerControl:
    """Window to control the lasers of the combiner: ON/OFF, power gauge (left : %, right : mW), manual value."""

    GAUGE_H = 200          # height of the gauge (pixels)
    GAUGE_MARGIN = 10      # margin above and below the gauge
    TROUGH_X = (42, 58)    # left and right of the trough
    CANVAS_W = 110
    TICKS = (0, 25, 50, 75, 100)   # (%)

    def __init__(self, combi: Combiner, x: Optional[int] = None, y: int = 0):
        self.combi = combi
        self.widgets = {}      # {wavelength: widgets and displayed values of the laser}

        self.root = tk.Tk()
        self.root.title(str(combi.model) + ', id = ' + str(combi.serial_number))
        self.root.attributes('-topmost', True)
        self.root.resizable(False, False)
        if x is not None:
            self.root.geometry('+' + str(int(x)) + '+' + str(int(y)))

        frame = tk.Frame(self.root, padx = 5, pady = 5)
        frame.pack()
        for col, (wavelength, info) in enumerate(combi.lasers.items()):
            self._build_laser(frame, col, wavelength, info['max_power'])

        tk.Button(self.root, text = 'CLOSE', command = self.root.destroy).pack(pady = (0, 8), fill = 'x', padx = 8)

        for wavelength in self.widgets:
            self._read(wavelength)
        self.root.after(1000, self._refresh)

    # ----- construction -----
    def _build_laser(self, parent, col, wavelength, max_power):
        color = wavelength_to_color(wavelength)
        box = tk.LabelFrame(parent, text = f"{wavelength:g} nm", font = ('Arial', 10, 'bold'), padx = 3, pady = 3)
        box.grid(row = 0, column = col, padx = 3)

        entry = tk.Entry(box, width = 8, justify = 'center')
        entry.pack()
        entry.bind('<Return>', lambda event, w = wavelength: self._entry_apply(w))
        tk.Label(box, text = "% or 'x mW'", font = ('Arial', 7), fg = 'gray').pack()

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
            if self.root.focus_get() is not wid['entry']:   # do not erase a value being typed
                wid['entry'].delete(0, 'end')
                wid['entry'].insert(0, f"{wid['percent']:.3g} %")
        if state is not None:
            wid['state'] = state
            color = wavelength_to_color(wavelength)
            wid['button'].config(text = 'ON' if state else 'OFF', bg = color if state else '#d0d0d0',
                                 fg = text_color(color) if state else 'black', activebackground = color if state else '#e0e0e0',
                                 activeforeground = text_color(color) if state else 'black')

    def _read(self, wavelength):
        """Read the power and the state on the laser."""
        percent, state = self.combi.read_laser(wavelength)
        self._show(wavelength, percent, state)

    def _refresh(self):
        """Display the last known values (no serial communication), in case they are changed from the console."""
        try:
            for wavelength, wid in self.widgets.items():
                if wid['dragging']:
                    continue
                laser = self.combi._laser(wavelength)
                state = None if getattr(laser, 'ttl_port', None) is not None else getattr(laser, 'output', None)
                self._show(wavelength, getattr(laser, 'power_pourcent', None), state)
            self.root.after(1000, self._refresh)
        except Exception:   # combiner disconnected
            pass

    # ----- actions -----
    def _set_power(self, wavelength, power, unit = '%'):
        try:
            self.combi.set_power(wavelength, *self.combi.clamp_power(wavelength, power, unit))
        except Exception as e:
            messagebox.showerror('Error', f"Laser {wavelength:g} nm, power not set : {e}", parent = self.root)
        self._read(wavelength)

    def _drag(self, wavelength, event):
        self.widgets[wavelength]['dragging'] = True
        percent = (self.GAUGE_MARGIN + self.GAUGE_H - event.y) / self.GAUGE_H * 100
        self._show(wavelength, percent)

    def _drag_apply(self, wavelength):
        wid = self.widgets[wavelength]
        wid['dragging'] = False
        self._set_power(wavelength, round(wid['percent'], 1), '%')

    def _entry_apply(self, wavelength):
        wid = self.widgets[wavelength]
        text = wid['entry'].get().strip().lower().replace(',', '.')
        unit = 'mW' if text.endswith('mw') else '%'
        try:
            value = float(text.replace('mw', '').replace('%', '').strip())
        except ValueError:
            messagebox.showerror('Error', "Enter a number : '20' for 20 % or '20 mW'", parent = self.root)
            return
        max_power = wid['max_power']
        if value < 0 or (unit == '%' and value > 100) or (unit == 'mW' and max_power is not None and value > max_power):
            messagebox.showerror('Error', f"Laser {wavelength:g} nm : value out of range (0 - 100 % or 0 - {max_power} mW)", parent = self.root)
            return
        self.root.focus_set()   # leave the box : it is updated with the value read on the laser
        self._set_power(wavelength, value, unit)

    def _toggle(self, wavelength):
        try:
            if self.widgets[wavelength]['state']:
                self.combi.laser_off(wavelength)
            else:
                self.combi.laser_on(wavelength)
        except Exception as e:
            messagebox.showerror('Error', f"Laser {wavelength:g} nm : {e}", parent = self.root)
        self._read(wavelength)

    def run(self):
        self.root.mainloop()
