# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:15:17 2026

@author: equipe-onli
"""
import time
import clr  # Pour utiliser les DLL Thorlabs (si nécessaire)
clr.AddReference("Thorlabs.MotionControl.FilterFlipper.dll")  # À décommenter si tu utilises la DLL Thorlabs

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.FilterFlipperCLI import FilterFlipper

class MFF:
    def __init__(self, serial_number: str):
        """
        Initialise la communication avec le MFF101 via son numéro de série.

        Args:
            serial_number (str): Numéro de série de l'appareil.
        """
        self.serial_number = serial_number
        self.device = None  # À remplacer par l'objet de communication réel (ex: via DLL Thorlabs)
        self.is_polling = False
        self.polling_delay = 0.1  # Délai par défaut en secondes
        self._init_communication()

    def _init_communication(self) -> None:
        """ Initialise la communication avec l'appareil. """
        try:
            # Exemple : Initialisation via la DLL Thorlabs (à adapter selon ta configuration)
            # self.device = Thorlabs.MotionControl.FilterFlipper.FilterFlipper.CreateFilterFlipper(self.serial_number)
            
            DeviceManagerCLI.BuildDeviceList()
            self.device = FilterFlipper.CreateFilterFlipper(self.serial)
            self.device.Connect(self.serial_number)
            print(f"Communication initialisée avec le MFF101 (SN: {self.serial_number}).")
        except Exception as e:
            print(f"Erreur lors de l'initialisation : {e}")

    def get_position(self) -> int:
        """
        Lit la position actuelle du miroir (1 ou 2).

        Returns:
            int: Position actuelle (1 ou 2).
        """
        try:
            # Exemple : position = self.device.Position  # À adapter selon la DLL
            position = 1  # Valeur par défaut pour l'exemple
            print(f"Position actuelle : {position}")
            return position
        except Exception as e:
            print(f"Erreur lors de la lecture de la position : {e}")
            return -1  # Valeur d'erreur

    def set_polling_delay(self, delay: float) -> None:
        """
        Définit le délai de polling (en secondes).

        Args:
            delay (float): Délai en secondes.
        """
        self.polling_delay = delay
        print(f"Polling delay défini à {delay} secondes.")

    def move_to_position(self, target_position: int) -> None:
        """
        Déplace le miroir vers la position souhaitée (1 ou 2).
        Vérifie d'abord la position actuelle pour éviter un changement inutile.

        Args:
            target_position (int): Position cible (1 ou 2).
        """
        if target_position not in [1, 2]:
            print("Erreur : La position doit être 1 ou 2.")
            return

        current_position = self.get_position()
        if current_position == target_position:
            print(f"Le miroir est déjà en position {target_position}.")
            return

        try:
            print(f"Changement de position vers {target_position}...")
            # Exemple : self.device.MoveTo(target_position)  # À adapter selon la DLL
            time.sleep(1)  # Attendre le temps nécessaire pour le mouvement (à ajuster)
            print(f"Miroir en position {target_position}.")
        except Exception as e:
            print(f"Erreur lors du changement de position : {e}")

    def close_communication(self) -> None:
        """ Ferme la communication avec l'appareil. """
        try:
            # Exemple : self.device.Disconnect(True)  # À adapter selon la DLL
            print("Communication fermée.")
        except Exception as e:
            print(f"Erreur lors de la fermeture de la communication : {e}")