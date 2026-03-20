# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 16:18:31 2026

@author: mahieu
"""

# -*- coding: utf-8 -*-
from pipython import GCSDevice, pitools
import tkinter as tk
import threading
import numpy as np  
from typing import Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import time

class stage:
    def __init__(self):

        self.pidevice = None
        self.stage_tools = None
    
    

def init_PI(Model: str = 'C-884', SN: str = '0000000000', verbose: bool = False):
    """
    Initialize the PI translation stage (X, Y, Z)
    
    Parameters
    ----------
    Model : str, optional
        the model of the PI translation stage. 
        The default is 'C-884'.
    SN : str, optional
        The serial number. 
        The default is '0000000000'.

    Returns
    -------
    the pitools object.

    """

    pidevice = GCSDevice(Model)
    stage_tools = pitools
    
    try:
        # On définit les modèles pour chaque axe (Axe 3 inclus ici suite à votre test)
        stages_model = ['M-111.1DG', 'M-111.1DG', 'M-111.1DG'] 

        # 1. Connexion au contrôleur
        pidevice = GCSDevice(Model)
        pidevice.ConnectUSB(serialnum=SN)
        if verbose:
            print(f"Connecté au {pidevice.qIDN().strip()}")
        # check the available axes
        axes = pitools.getaxeslist(pidevice, None)
        # 2. Assignation des platines (CST)
        # Indispensable pour que le contrôleur sache quel moteur il pilote
        if verbose:
            print("Assignation des platines...")
        pidevice.CST(axes, stages_model)

        # 3. Activation du Servo (SVO)
        # Les moteurs DC comme le M-111.1DG ne bougent pas si le servo est sur False
        if verbose:
            print("Activation des servos...")
        pidevice.SVO(axes, [True, True, True])

        # 4. Référencement (Homing)
        # On lance la recherche de la limite négative (FNL) pour calibrer le zéro
        if verbose:
            print("Référencement en cours (Homing)...")
        pidevice.FNL(axes)

        # 5. Attente de la fin du référencement
        # Très important : bloque le code jusqu'à ce que les axes soient à l'arrêt et référencés
        pitools.waitontarget(pidevice, axes)
        if verbose:
            print("Tous les axes sont référencés et prêts.")
        
        print('PI tanslation stage connected')
        
    except:
        print('!!!! Warning, PI tanslation stage not connected !!!!')
        print('Turn on the controller and wait until the both green led are stabilized')
        
    # return pidevice, stage_tools
    stage.pidevice = pidevice
    stage.stage_tools = stage_tools
    return stage


@dataclass_json
@dataclass
class stage_parameters:
    """
    
    """
    array_to_move: Optional[float] = None
    class_description: str = None
    
    def __post_init__(self, array_to_move):
        self.array_to_move = array_to_move
        self.class_description = 'Stage translation PI parameters' 


def read_position(pidevice: object, stage_tools: object, axes_xyz: list = ['X', 'Y', 'Z'], verbose: bool = False):
    """
    Read the position of the three axes

    Parameters
    ----------
    pidevice: object.
        to parameter (servo, referencement, ...) the stages
    stage_tools : object
        Contain all the commands of the stage.
    axes_xyz : list.
        A list of the axes to read their position.
        Explanation: The object device call the axes 1, 2 or 3. We want a correspondance with the letters X, Y, Z. 

    Returns
    -------
    position: list.
        return a list of the position of the whished axes.

    """
    axes = stage_tools.getaxeslist(pidevice, None)
    
    position = []
    for i in axes:        
        pos_dict = pidevice.qPOS(i) # Current position
        pos = list(pos_dict.values())[0]
        if verbose:
            print('Current position of ' + axes_xyz[int(i)-1] + ' = ' + str(pos) + ' mm')
        position.append(pos)

    return position


def move_to_middle(pidevice, stage_tools):
    # On ajoute un flag 'done' pour la surveillance
    status = {"stop": False, "done": False}

    def emergency_stop():
        status["stop"] = True
        try:
            stage_tools.stopall(pidevice)
            print("ARRÊT D'URGENCE : Commandes envoyées.")
        except Exception as e:
            print(f"Erreur lors du stop : {e}")
        status["done"] = True # Force la sortie du polling

    root = tk.Tk()
    root.title("Contrôle Mouvement")
    root.attributes("-topmost", True)

    tk.Label(root, text="MOUVEMENT EN COURS", font=("Arial", 12)).pack(pady=10)
    tk.Button(root, text="STOP", bg="red", fg="white", font=("Arial", 14, "bold"), 
              command=emergency_stop).pack(pady=10)

    # --- NOUVEAU : Fonction de surveillance (Polling) ---
    def check_status():
        if status["done"]:
            root.quit() # Arrête le mainloop immédiatement
        else:
            # On se ré-appelle dans 100ms pour garder la boucle "éveillée"
            root.after(100, check_status)

    def motion_thread():
        try:
            axes = stage_tools.getaxeslist(pidevice, None)
            stage_tools.movetomiddle(pidevice, axes)

            while not status["stop"]:
                moving_states = pidevice.IsMoving(axes).values()
                if not any(moving_states):
                    break
                time.sleep(0.1)
            
            print("Mouvement terminé.")
        except Exception as e:
            print(f"Erreur thread : {e}")
        finally:
            status["done"] = True # Signale à la GUI qu'on a fini

    # On lance la surveillance AVANT le mainloop
    root.after(100, check_status)
    
    t = threading.Thread(target=motion_thread, daemon=True)
    t.start()

    root.mainloop()
    
    # Nettoyage final
    try:
        root.destroy()
    except:
        pass

  
def move_an_axis(pidevice: object, stage_tools: object, axes: list = ['2'], array_to_move: np.array = np.empty(0), verbose: bool = False):
    """
    Move the three stages to the middle

    Parameters
    ----------
    pidevice: object.
        to parameter (servo, referencement, ...) the stages
    stage_tools : object
        Contain all the commands of the stage.
    array_to_move, numpy array.
        the array to move one axis. The value inside the array are absolute values

    Returns
    -------
    None.

    """
    maxrange_dict = pitools.getmaxtravelrange(pidevice, axes)
    maxrange = list(maxrange_dict.values())[0]

    if max(array_to_move) > maxrange:
        print('Warning, array_to_move is greater than the maximum travel range, please decrase the maximum value below: ' + 
              str(maxrange) + ' or above "0"')
    else:
        for i in array_to_move:
            pidevice.MOV(axes, i)
            stage_tools.waitontarget(pidevice, None)
            if verbose:
                print('stage on target: ' + str(i) + ' mm')
    


def go_to_zero(pidevice: object, stage_tools: object):
    """
    send the 3 axes to zero

    Parameters
    ----------
    pidevice: object.
        to parameter (servo, referencement, ...) the stages
    stage_tools : object
        Contain all the commands of the stage.

    Returns
    -------
    None.

    """
    axes = stage_tools.getaxeslist(pidevice, None)
    for i in axes: 
        pidevice.MOV(i, 0)
        
    stage_tools.waitontarget(pidevice, None)
    print('all axes are set to zero')
    

# import tkinter as tk
from tkinter import messagebox

class PIJogControl:
    def __init__(self, pidevice):
        self.pidevice = pidevice
        self.axes = ['1', '2', '3']
        
        # Récupération des limites et positions actuelles
        self.min_limits = self.pidevice.qTMN(self.axes)
        self.max_limits = self.pidevice.qTMX(self.axes)
        
        self.root = tk.Tk()
        self.root.title("Contrôle Manuel PI C-884")
        self.root.attributes("-topmost", True)

        # Variable pour le pas de déplacement
        self.step_var = tk.DoubleVar(value=0.1)
        self.pos_labels = {}

        self._build_ui()
        self._update_positions()

    def _build_ui(self):
        # Section Pas de déplacement
        step_frame = tk.Frame(self.root, pady=10)
        step_frame.pack()
        tk.Label(step_frame, text="Pas (mm):").grid(row=0, column=0)
        tk.Entry(step_frame, textvariable=self.step_var, width=10).grid(row=0, column=1)

        # Section Axes
        axes_frame = tk.Frame(self.root, padx=20, pady=10)
        axes_frame.pack()

        for i, axis in enumerate(self.axes):
            tk.Label(axes_frame, text=f"Axe {axis}", font=('Arial', 10, 'bold')).grid(row=i, column=0, padx=10)
            
            # Affichage Position
            self.pos_labels[axis] = tk.Label(axes_frame, text="0.000", fg="blue", font=('Consolas', 12))
            self.pos_labels[axis].grid(row=i, column=1, padx=10)

            # Boutons de mouvement
            tk.Button(axes_frame, text=" ◀ ", command=lambda a=axis: self.move(a, -1)).grid(row=i, column=2, pady=5)
            tk.Button(axes_frame, text=" ▶ ", command=lambda a=axis: self.move(a, 1)).grid(row=i, column=3, pady=5)

        # Bouton Fermer
        tk.Button(self.root, text="FINIR & FERMER", bg="#4CAF50", fg="white", 
                  command=self.root.destroy).pack(pady=20, fill='x')

    def _update_positions(self):
        """Met à jour l'affichage des positions réelles."""
        positions = self.pidevice.qPOS(self.axes)
        for axis, pos in positions.items():
            self.pos_labels[axis].config(text=f"{pos:.4f} mm")
        # On rafraîchit toutes les 500ms au cas où un mouvement est lent
        self.root.after(500, self._update_positions)

    def move(self, axis, direction):
        try:
            step = self.step_var.get()
            current_pos = self.pidevice.qPOS(axis)[axis]
            new_pos = current_pos + (step * direction)

            # Vérification des bornes
            if new_pos < self.min_limits[axis]:
                new_pos = self.min_limits[axis]
                print(f"Limite MIN atteinte sur l'axe {axis}")
            elif new_pos > self.max_limits[axis]:
                new_pos = self.max_limits[axis]
                print(f"Limite MAX atteinte sur l'axe {axis}")

            # Commande de mouvement
            self.pidevice.MOV(axis, new_pos)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Mouvement impossible : {e}")

    def run(self):
        self.root.mainloop()

# --- Utilisation ---
# jog = PIJogControl(pidevice)
# jog.run()


def stage_adjustment(pidevice):
    # On crée une fonction interne qui servira de cible au thread
    def start_gui():
        gui = PIJogControl(pidevice)
        gui.run() # Le mainloop tournera ici, dans son propre thread

    # Création du thread
    # daemon=True permet à la fenêtre de se fermer si vous quittez Python
    jog_thread = threading.Thread(target=start_gui, daemon=True)
    jog_thread.start()
    # print("Fenêtre de contrôle lancée en arrière-plan. Vous avez la main !")
    
    
def disconnect_stage(stage: object, go_home: bool = True):
    """
    disconnect the PI translation stage

    Parameters
    ----------
    stage : object
        Contain all the commands of the stage.
    go_home: bool.
        set the three stage to zero before disconnection
    Returns
    -------
    None.

    """
    if go_home:
        go_to_zero(stage.pidevice, stage.stage_tools)
        
    stage.pidevice.CloseConnection()
    print('PI tanslation stage disconnected')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    