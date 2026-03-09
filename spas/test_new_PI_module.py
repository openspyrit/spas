# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:46:19 2026

@author: equipe-onli
"""
# -*- coding: utf-8 -*-
"""
PI translation stage controller module
Refactored object-oriented version
"""

from pipython import GCSDevice, pitools
import tkinter as tk
import threading
import numpy as np
import time
from tkinter import messagebox
import multiprocessing


class PIStage:
    """
    Controller for PI translation stages (C-884 + M-111.1DG)
    """

    def __init__(self, Model='C-884', SN='0000000000', verbose=False):

        self.Model = Model
        self.SN = SN
        self.verbose = verbose

        self.pidevice = None
        self.tools = pitools

        self._connect()


    # ------------------------------------------------------------------
    # CONNECTION
    # ------------------------------------------------------------------

    def _connect(self):

        try:

            stages_model = ['M-111.1DG', 'M-111.1DG', 'M-111.1DG']

            self.pidevice = GCSDevice(self.Model)
            self.pidevice.ConnectUSB(serialnum=self.SN)

            if self.verbose:
                print(f"Connected to {self.pidevice.qIDN().strip()}")

            self.axes = self.tools.getaxeslist(self.pidevice, None)

            if self.verbose:
                print("Assigning stages...")

            self.pidevice.CST(self.axes, stages_model)

            if self.verbose:
                print("Activating servos...")

            self.pidevice.SVO(self.axes, [True, True, True])

            if self.verbose:
                print("Referencing axes (Homing)...")

            self.pidevice.FNL(self.axes)

            self.tools.waitontarget(self.pidevice, self.axes)

            print("PI translation stage connected")

        except Exception:

            print("!!!! Warning, PI translation stage not connected !!!!")
            print("Turn on the controller and wait until the green LEDs stabilize")


    # ------------------------------------------------------------------
    # POSITION
    # ------------------------------------------------------------------

    def read_position(self, axes_xyz=['X', 'Y', 'Z'], verbose=False):

        position = []

        for i in self.axes:

            pos_dict = self.pidevice.qPOS(i)
            pos = list(pos_dict.values())[0]

            if verbose:
                print(f"Current position of {axes_xyz[int(i)-1]} = {pos} mm")

            position.append(pos)

        return position


    # ------------------------------------------------------------------
    # MOVE AXIS
    # ------------------------------------------------------------------

    def move_axis(self, axis='2', array_to_move=np.empty(0), verbose=False):

        maxrange_dict = pitools.getmaxtravelrange(self.pidevice, [axis])
        maxrange = list(maxrange_dict.values())[0]

        if max(array_to_move) > maxrange:

            print(
                "Warning: array_to_move exceeds max travel range "
                + str(maxrange)
            )

        else:
            print("Move requested", axis)
            for i in array_to_move:

                self.pidevice.MOV(axis, i)
                # self.stage.pidevice.MOV(axis, i)
                self.tools.waitontarget(self.pidevice, None)

                if verbose:
                    print(f"stage on target: {i} mm")


    # ------------------------------------------------------------------
    # MOVE TO ZERO
    # ------------------------------------------------------------------

    def go_to_zero(self):

        for axis in self.axes:
            self.pidevice.MOV(axis, 0)

        self.tools.waitontarget(self.pidevice, None)

        print("All axes set to zero")


    # ------------------------------------------------------------------
    # MOVE TO MIDDLE
    # ------------------------------------------------------------------

    def move_to_middle(self):

        status = {"stop": False, "done": False}

        def emergency_stop():

            status["stop"] = True

            try:
                self.tools.stopall(self.pidevice)
                print("EMERGENCY STOP sent")

            except Exception as e:
                print(f"Stop error: {e}")

            status["done"] = True


        root = tk.Tk()
        root.title("Motion control")
        root.attributes("-topmost", True)

        tk.Label(root, text="MOTION IN PROGRESS",
                 font=("Arial", 12)).pack(pady=10)

        tk.Button(root,
                  text="STOP",
                  bg="red",
                  fg="white",
                  font=("Arial", 14, "bold"),
                  command=emergency_stop).pack(pady=10)


        def check_status():

            if status["done"]:
                root.quit()
            else:
                root.after(100, check_status)


        def motion_thread():

            try:

                self.tools.movetomiddle(self.pidevice, self.axes)

                while not status["stop"]:

                    moving_states = self.pidevice.IsMoving(self.axes).values()

                    if not any(moving_states):
                        break

                    time.sleep(0.1)

                print("Motion finished")

            except Exception as e:
                print(f"Thread error: {e}")

            finally:
                status["done"] = True


        root.after(100, check_status)

        t = threading.Thread(target=motion_thread, daemon=True)
        t.start()

        root.mainloop()

        try:
            root.destroy()
        except:
            pass


    # ------------------------------------------------------------------
    # MANUAL JOG CONTROL
    # ------------------------------------------------------------------

    def stage_adjustment(self):

        def start_gui():
            gui = PIJogControl(self)
            gui.run()
    
        self.jog_thread = threading.Thread(
            target=start_gui,
            daemon=True
        )
    
        self.jog_thread.start()

    # def stage_adjustment(self):

    #     def start_gui():
    #         gui = PIJogControl(self.pidevice)
    #         gui.run()
    
    #     jog_process = multiprocessing.Process(
    #         target=start_gui,
    #         daemon=True
    #     )
    
    #     jog_process.start()
    
    # def stage_adjustment(self):

    #     def start_gui():
    #         gui = PIJogControl(self.pidevice)
    #         gui.run()

    #     jog_thread = threading.Thread(target=start_gui, daemon=True)
    #     jog_thread.start()


    # ------------------------------------------------------------------
    # DISCONNECT
    # ------------------------------------------------------------------

    def disconnect(self, go_home=True):

        if go_home:
            self.go_to_zero()

        self.pidevice.CloseConnection()

        print("PI translation stage disconnected")


# ----------------------------------------------------------------------
# JOG GUI
# ----------------------------------------------------------------------

class PIJogControl:

    # def __init__(self, pidevice):
    def __init__(self, stage):

        # self.pidevice = pidevice
        self.stage = stage
        self.pidevice = stage.pidevice
        self.axes = stage.axes
        # self.axes = ['1', '2', '3']

        self.min_limits = self.pidevice.qTMN(self.axes)
        self.max_limits = self.pidevice.qTMX(self.axes)

        self.root = tk.Tk()
        self.root.title("PI Manual Control")
        # self.root.attributes("-topmost", True)
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(500, lambda: self.root.attributes("-topmost", False))
        
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        
        # self.root.lift()
        # self.root.attributes('-topmost', True)
        # self.root.after_idle(self.root.attributes, '-topmost', False)

        self.step_var = tk.DoubleVar(value=0.1)
        self.pos_labels = {}

        self._build_ui()
        self._update_positions()


    def _build_ui(self):

        step_frame = tk.Frame(self.root, pady=10)
        step_frame.pack()

        tk.Label(step_frame, text="Step (mm):").grid(row=0, column=0)
        tk.Entry(step_frame,
                 textvariable=self.step_var,
                 width=10).grid(row=0, column=1)

        axes_frame = tk.Frame(self.root, padx=20, pady=10)
        axes_frame.pack()

        for i, axis in enumerate(self.axes):

            tk.Label(axes_frame,
                     text=f"Axis {axis}",
                     font=('Arial', 10, 'bold')).grid(row=i, column=0)

            self.pos_labels[axis] = tk.Label(
                axes_frame,
                text="0.000",
                fg="blue",
                font=('Consolas', 12)
            )

            self.pos_labels[axis].grid(row=i, column=1)

            tk.Button(axes_frame,
                      text=" ◀ ",
                      command=lambda a=axis: self.move(a, -1)
                      ).grid(row=i, column=2)

            tk.Button(axes_frame,
                      text=" ▶ ",
                      command=lambda a=axis: self.move(a, 1)
                      ).grid(row=i, column=3)

        tk.Button(self.root,
                  text="FINISH & CLOSE",
                  bg="#4CAF50",
                  fg="white",
                  command=self.root.destroy).pack(pady=20, fill='x')


    def _update_positions(self):

        positions = self.pidevice.qPOS(self.axes)

        for axis, pos in positions.items():
            self.pos_labels[axis].config(text=f"{pos:.4f} mm")

        self.root.after(500, self._update_positions)


    def move(self, axis, direction):

        try:

            step = self.step_var.get()

            current_pos = self.pidevice.qPOS(axis)[axis]
            new_pos = current_pos + (step * direction)

            if new_pos < self.min_limits[axis]:
                new_pos = self.min_limits[axis]

            elif new_pos > self.max_limits[axis]:
                new_pos = self.max_limits[axis]

            self.pidevice.MOV(axis, new_pos)

        except Exception as e:

            messagebox.showerror("Error", f"Motion impossible: {e}")


    def run(self):
        self.root.mainloop()