# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 16:18:31 2026

@author: mahieu

Control of the PI translation stages (controller C-884 + 3 stages M-111.1DG) with the PI package "pipython".
The previous version, with functions, is in PI_module_functions.py

Example:
    stage = PIStage()
    stage.init(model = 'C-884', SN = '0000000000', verbose = True)
    stage.move_to_middle()
    position = stage.read_position(verbose = True)
    stage.move_axis(axis = '2', array_to_move = [5.0], verbose = True)
    stage.stage_adjustment()
    stage.disconnect(go_home = True)
"""

from pipython import GCSDevice, pitools
import tkinter as tk
from tkinter import messagebox
import os
import sys
import json
import tempfile
import threading
import subprocess
from multiprocessing.connection import Listener, Client
import numpy as np
from typing import Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import time


class PIStage:
    """Class that controls the PI translation stages (X, Y, Z).

    Attributes:
        pidevice (GCSDevice):
            the PI object of the controller. Can be used for the GCS commands not implemented here.
        tools (module):
            pitools, the PI functions (waitontarget, movetomiddle, stopall...).
        model (str):
            the model of the controller.
        serial_number (str):
            the serial number of the controller.
        axes (list):
            the axes of the controller ('1', '2', '3').
        axes_names (list):
            the names of the axes, in the same order: 'X', 'Y', 'Z'.
    """

    def __init__(self):
        self.pidevice = None
        self.tools = pitools
        self.model = None
        self.serial_number = None
        self.axes = []
        self.axes_names = ['X', 'Y', 'Z']
        self._gui = None        # process of the manual control window (subprocess.Popen)
        self._gui_conn = None   # connection with the manual control window


    def init(self, model: str = 'C-884', SN: str = '0000000000', stages_model: Optional[list] = None, verbose: bool = False):
        """
        Initialize the PI translation stage (X, Y, Z): connection, stages assignment, servo and referencing (homing).

        Parameters
        ----------
        model : str, optional
            the model of the controller. The default is 'C-884'.
        SN : str, optional
            The serial number of the controller. The default is '0000000000'.
        stages_model : list, optional
            the model of the stage of each axis. The default is None : 'M-111.1DG' for all the axes.
        verbose : bool, optional
            print the steps of the initialization. The default is False.
        """
        try:
            # 1. Connection to the controller
            self.pidevice = GCSDevice(model)
            self.pidevice.ConnectUSB(serialnum=SN)
            if verbose:
                print(f"Connected to {self.pidevice.qIDN().strip()}")
            # check the available axes
            self.axes = self.tools.getaxeslist(self.pidevice, None)
            # 2. Assignment of the stages (CST)
            # Required so that the controller knows which motor it drives
            if stages_model is None:
                stages_model = ['M-111.1DG'] * len(self.axes)
            if verbose:
                print("Assigning the stages...")
            self.pidevice.CST(self.axes, stages_model)
            # 3. Activation of the servo (SVO)
            # DC motors such as the M-111.1DG do not move if the servo is False
            if verbose:
                print("Activating the servos...")
            self.pidevice.SVO(self.axes, [True] * len(self.axes))
            # 4. Referencing (homing)
            # Search of the negative limit (FNL) to calibrate the zero
            if verbose:
                print("Referencing the axes (homing)...")
            self.pidevice.FNL(self.axes)
            # 5. Wait for the end of the referencing
            # Very important: blocks the code until the axes are stopped and referenced
            self.tools.waitontarget(self.pidevice, self.axes)
            if verbose:
                print("All the axes are referenced and ready.")
        except Exception as e:
            raise RuntimeError('PI translation stage not connected (' + str(e) + '). '
                               'Turn on the controller and wait until the both green led are stabilized') from e

        self.model = model
        self.serial_number = SN
        print('PI translation stage ' + model + ' connected')


    def read_position(self, verbose: bool = False) -> list:
        """
        Read the position of the three axes

        Parameters
        ----------
        verbose : bool, optional
            print the positions. The default is False.

        Returns
        -------
        position: list.
            the position (mm) of each axis, in the order of self.axes.
        """
        positions = self.pidevice.qPOS(self.axes)
        position = []
        for i, axis in enumerate(self.axes):
            if verbose:
                print('Current position of ' + self.axes_names[i] + ' = ' + str(positions[axis]) + ' mm')
            position.append(positions[axis])

        return position


    def move_axis(self, axis: str = '2', array_to_move = np.empty(0), verbose: bool = False):
        """
        Move one axis successively to each position of array_to_move

        Parameters
        ----------
        axis : str
            the axis to move ('1', '2' or '3'). The default is '2' (Y).
        array_to_move : list or numpy array.
            the positions (mm) to reach. The values are absolute values.
        verbose : bool, optional
            print each position reached. The default is False.
        """
        array_to_move = np.atleast_1d(array_to_move)
        min_range = self.pidevice.qTMN(axis)[axis]
        max_range = self.pidevice.qTMX(axis)[axis]

        if np.max(array_to_move) > max_range or np.min(array_to_move) < min_range:
            print('Warning, array_to_move is out of the travel range of the axis ' + axis + ', the values must be between ' +
                  str(min_range) + ' and ' + str(max_range) + ' mm. The stage does not move.')
            return

        for position in array_to_move:
            position = float(position)
            self.pidevice.MOV(axis, position)
            self.tools.waitontarget(self.pidevice, axis)
            if verbose:
                print('stage on target: ' + str(position) + ' mm')


    def go_to_zero(self):
        """send the 3 axes to zero"""
        for axis in self.axes:
            self.pidevice.MOV(axis, 0)

        self.tools.waitontarget(self.pidevice, self.axes)
        print('all axes are set to zero')


    def move_to_middle(self):
        """Move the 3 axes to the middle of their travel range, with a window to stop the motion."""
        # 'done' flag for the monitoring
        status = {"stop": False, "done": False}

        def emergency_stop():
            status["stop"] = True
            try:
                self.tools.stopall(self.pidevice)
                print("EMERGENCY STOP: commands sent.")
            except Exception as e:
                print(f"Error during the stop: {e}")
            status["done"] = True # Forces the exit of the polling

        root = tk.Tk()
        root.title("Motion control")
        root.attributes("-topmost", True)

        tk.Label(root, text="MOTION IN PROGRESS", font=("Arial", 12)).pack(pady=10)
        tk.Button(root, text="STOP", bg="red", fg="white", font=("Arial", 14, "bold"),
                  command=emergency_stop).pack(pady=10)

        # Monitoring function (polling)
        def check_status():
            if status["done"]:
                root.quit() # Stops the mainloop immediately
            else:
                # Called again in 100 ms to keep the loop "awake"
                root.after(100, check_status)

        def motion_thread():
            try:
                self.tools.movetomiddle(self.pidevice, self.axes)

                while not status["stop"]:
                    moving_states = self.pidevice.IsMoving(self.axes).values()
                    if not any(moving_states):
                        break
                    time.sleep(0.1)

                print("Motion finished.")
            except Exception as e:
                print(f"Error in the motion thread: {e}")
            finally:
                status["done"] = True # Tells the GUI that the motion is finished

        # The monitoring is started BEFORE the mainloop
        root.after(100, check_status)

        t = threading.Thread(target=motion_thread, daemon=True)
        t.start()

        root.mainloop()

        # Final cleaning
        try:
            root.destroy()
        except:
            pass


    def stage_adjustment(self, x: Optional[int] = None, y: int = 0, align: str = 'left'):
        """Open the window to move manually the axes : the console stays available.

        The window runs in its own Python process (tkinter in a thread of the kernel crashes it :
        "Tcl_AsyncDelete: async handler deleted by the wrong thread"). The controller stays in the kernel,
        a thread (without tkinter) sends the motions asked by the window to the stage and the positions
        to the window, through a local connection.

        Parameters
        ----------
        x, y : int, optional
            the position (pixels) of the top left corner of the window on the screen
            (of the top right corner if align = 'right').
            To place it on the right of the camera display: x = cam.display_width(curve = True) + 20.
            To place it on the right of the image, below the profile: x, y = cam.display_below_profile().
            The default is None : position chosen by Windows.
            If the window is already open, it is moved there (nothing done if x is None).
        align : str, optional
            'left' : x is the left edge of the window, 'right' : x is its right edge. The default is 'left'.
        """
        if self.pidevice is None:
            print('PI translation stage not connected, run stage.init() first')
            return
        if self._gui is not None and self._gui.poll() is None:   # already open : only moved
            if x is not None and self._gui_conn is not None:
                try:
                    self._gui_conn.send(('move', int(x), int(y), align))
                except Exception:
                    pass
            return

        key = os.urandom(16)
        listener = Listener(('127.0.0.1', 0), authkey = key)   # local connection with the window
        settings = {'port': listener.address[1], 'title': 'PI manual control ' + str(self.model),
                    'axes': list(self.axes), 'x': x, 'y': y, 'align': align}
        env = dict(os.environ, PI_WINDOW_KEY = key.hex())
        log = tempfile.NamedTemporaryFile('w+', prefix = 'PI_window_', suffix = '.log', delete = False)
        self._gui = subprocess.Popen([sys.executable, '-c', 'from spas.PI_module import _jog_window_main; _jog_window_main()',
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
            text = log.read().strip()
            print('Manual control window of the PI stage : not opened.' + ('\n' + text if text else ''))
            if process.poll() is None:
                process.kill()
            return None
        finally:
            listener.close()

    def _serve_window(self, listener, log):
        """Thread of the kernel (no tkinter) : moves the axes asked by the window ('step', axis, direction, step)
        and sends it the positions every 500 ms."""
        conn = self._connect_window(listener, log)
        if conn is None:
            return
        self._gui_conn = conn
        try:
            # Travel range limits of the axes
            min_limits = self.pidevice.qTMN(self.axes)
            max_limits = self.pidevice.qTMX(self.axes)
            last_update = 0
            while self.pidevice is not None:
                if conn.poll(0.1):
                    message = conn.recv()
                    if message[0] == 'closed':
                        break
                    if message[0] == 'step':
                        axis, direction, step = message[1:]
                        try:
                            new_pos = self.pidevice.qPOS(axis)[axis] + step * direction
                            # Check of the limits
                            if new_pos < min_limits[axis]:
                                new_pos = min_limits[axis]
                                print(f"MIN limit reached on the axis {axis}")
                            elif new_pos > max_limits[axis]:
                                new_pos = max_limits[axis]
                                print(f"MAX limit reached on the axis {axis}")
                            # Motion command
                            self.pidevice.MOV(axis, new_pos)
                        except Exception as e:
                            conn.send(('error', f"Motion impossible: {e}"))
                # Refreshed every 500 ms, in case of a slow motion
                if time.time() - last_update > 0.5 and self.pidevice is not None:
                    conn.send(('positions', dict(self.pidevice.qPOS(self.axes))))
                    last_update = time.time()
        except (EOFError, OSError):   # window closed or crashed
            pass
        except Exception as e:
            if self.pidevice is not None:
                print('Manual control window of the PI stage : communication stopped (' + str(e) + ')')
        finally:
            conn.close()
            self._gui_conn = None

    def _close_window(self):
        """Close the manual control window, if it is open."""
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


    def disconnect(self, go_home: bool = True):
        """
        disconnect the PI translation stage

        Parameters
        ----------
        go_home: bool.
            set the three stage to zero before disconnection. The default is True.
        """
        self._close_window()
        if go_home:
            self.go_to_zero()

        self.pidevice.CloseConnection()
        self.pidevice = None
        print('PI translation stage disconnected')


@dataclass_json
@dataclass
class stage_parameters:
    """
    Class containing the parameters of the PI translation stage for the metadata.
    """
    array_to_move: Optional[list] = None
    class_description: str = 'Stage translation PI parameters'


def _jog_window_main():
    """Entry point of the process of the manual control window (tkinter in its main thread), launched by
    PIStage.stage_adjustment() : settings in argv (json), key of the connection in the environment."""
    settings = json.loads(sys.argv[1])
    conn = Client(('127.0.0.1', settings['port']), authkey = bytes.fromhex(os.environ['PI_WINDOW_KEY']))
    PIJogControl(conn, settings['axes'], settings['title'], settings['x'], settings['y'] or 0, settings['align']).run()


class PIJogControl:
    """Window to move manually the axes of the PI stage step by step.
    It runs in its own process and communicates with the kernel (PIStage._serve_window) through conn :
    it sends ('step', axis, direction, step), ('closed',) and receives ('positions', {axis: position}),
    ('error', text), ('move', x, y, align), ('close',)."""

    def __init__(self, conn, axes: list, title: str, x: Optional[int] = None, y: int = 0, align: str = 'left'):
        self.conn = conn
        self.axes = axes

        self.root = tk.Tk()
        self.root.title(title)
        self.root.attributes("-topmost", True)
        self._place(x, y, align)
        self.root.protocol('WM_DELETE_WINDOW', self.close)

        # Variable for the step of the motion
        self.step_var = tk.DoubleVar(value=0.1)
        self.pos_labels = {}

        self._build_ui()
        self.root.after(100, self._poll)

    def _place(self, x, y, align):
        if x is None:
            return
        if align == 'right':   # "-X" : X pixels between the right edge of the window and the right edge of the screen
            self.root.geometry('-' + str(self.root.winfo_screenwidth() - int(x)) + '+' + str(int(y)))
        else:
            self.root.geometry('+' + str(int(x)) + '+' + str(int(y)))

    def _build_ui(self):
        # Step section
        step_frame = tk.Frame(self.root, pady=10)
        step_frame.pack()
        tk.Label(step_frame, text="Step (mm):").grid(row=0, column=0)
        tk.Entry(step_frame, textvariable=self.step_var, width=10).grid(row=0, column=1)

        # Axes section
        axes_frame = tk.Frame(self.root, padx=20, pady=10)
        axes_frame.pack()

        for i, axis in enumerate(self.axes):
            tk.Label(axes_frame, text=f"Axis {axis}", font=('Arial', 10, 'bold')).grid(row=i, column=0, padx=10)

            # Position display
            self.pos_labels[axis] = tk.Label(axes_frame, text="0.000", fg="blue", font=('Consolas', 12))
            self.pos_labels[axis].grid(row=i, column=1, padx=10)

            # Motion buttons
            tk.Button(axes_frame, text=" ◀ ", command=lambda a=axis: self.move(a, -1)).grid(row=i, column=2, pady=5)
            tk.Button(axes_frame, text=" ▶ ", command=lambda a=axis: self.move(a, 1)).grid(row=i, column=3, pady=5)

        # Close button
        tk.Button(self.root, text="FINISH & CLOSE", bg="#4CAF50", fg="white",
                  command=self.close).pack(pady=20, fill='x')

    def _poll(self):
        """Messages from the kernel."""
        try:
            while self.conn.poll():
                message = self.conn.recv()
                if message[0] == 'positions':   # updates the display of the real positions
                    for axis, pos in message[1].items():
                        if axis in self.pos_labels:
                            self.pos_labels[axis].config(text=f"{pos:.4f} mm")
                elif message[0] == 'error':
                    messagebox.showerror("Error", message[1], parent=self.root)
                elif message[0] == 'move':
                    self._place(*message[1:])
                    self.root.lift()
                elif message[0] == 'close':
                    self.root.destroy()
                    return
        except (EOFError, OSError):   # kernel stopped
            self.root.destroy()
            return
        self.root.after(100, self._poll)

    def move(self, axis, direction):
        try:
            step = self.step_var.get()
        except tk.TclError:
            messagebox.showerror("Error", "Step: enter a number (mm)", parent=self.root)
            return
        try:
            self.conn.send(('step', axis, direction, step))
        except (EOFError, OSError):   # kernel stopped
            self.root.destroy()

    def close(self):
        try:
            self.conn.send(('closed',))
        except (EOFError, OSError):
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()
