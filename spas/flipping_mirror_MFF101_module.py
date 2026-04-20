# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:32:21 2026

@author: equipe-onli
"""
# -*- coding: utf-8 -*-

''' MFF101.py '''

"""
Début : Jeudi 28 janvier 2026
Auteur : Jo
Modifié par : Le Chat
"""

from time import sleep, time
from typing import Optional, List

from sys import path
from clr import AddReference

''' Erreurs '''
class RemoteError(Exception):
    """ Erreur renvoyée en cas de commande incorrecte. """
class InstrumentNotFoundError(Exception):
    """ Erreur indiquant qu'un harware instrument est introuvable. """

''' Config '''
SETUP_INSTR = {"TIMEOUT_S": 5, "TIMECOM_S": 0.05}
SETUP_MFF = {
    "DOC_NAME_USER": "mff101_user_guide.pdf",
    "DOC_NAME_PROG": "mff101_apt_communications_protocol.pdf",
    "POSITION_DELAY_S": 0.01,
    "POLLING_DELAY_DEFAULY": 250,
    "POLLING_DELAY_MIN": 20,
    "POLLING_DELAY_MAX": 3600
}
KINESIS_PATH = r"C:\Program Files\Thorlabs\Kinesis"
THORLABS = {
    "DEVICE_MANAGER": "Thorlabs.MotionControl.DeviceManagerCLI",
    "FILTER_FLIPPER": "Thorlabs.MotionControl.FilterFlipperCLI"
}

path.append(KINESIS_PATH)
AddReference(THORLABS["DEVICE_MANAGER"])
AddReference(THORLABS["FILTER_FLIPPER"])

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.FilterFlipperCLI import FilterFlipper

class MFF:
    """ Bibliothèque permettant de piloter les mirroirs Mff101 et Mff102"""
    def __init__(self, SN: str) -> None:
        self.serial = SN
        self._init_communication_settings()
        self._init_settings()
        self.open_communication()

    def _init_communication_settings(self) -> None:
        """ Déclaration des variables de communication. """
        self.device: FilterFlipper | None = None
        self._on: bool = False  # Etat de communication

    def _init_settings(self) -> None:
        """ Déclaration et initialisation des variables globales de contrôle. """
        self.idn: str | None = None  # Identification matériel
        self.polling_delay_ms: int = SETUP_MFF["POLLING_DELAY_DEFAULY"]  # Delay de com périodique par défaut
        self.polling_delay_min: int = SETUP_MFF['POLLING_DELAY_MIN']  # Min possible
        self.polling_delay_max: int = SETUP_MFF['POLLING_DELAY_MAX']  # Max possible
        self.is_polling: bool | None = None  # Fonction d'interrogation périodique du contrôlleur.
        self.ini_move_time: float = 0.0  # Temps initial si vérification de movement.
        self.cur_move_time: float = 0.0  # Temps courant si vérification de mouvement.
        self.target_position: int = 0

    def get_communication(self) -> bool:
        """
        Retourne l'état courant de la communication.
        -----------------------------------------------------
        Notes:
            Évite de manipuler directement la variable de contrôle.

        Returns:
            bool : État courant de la communication.
        """
        return self._on

    def open_communication(self) -> None:
        """
        Ouverture de la communication avec l'instrument.
        ----------------------------------------------
        Raises:
            InstrumentNotFoundError: Si instrument introuvable.
            FileNotFoundError: Si fichier DLL introuvable.
        """
        if self.get_communication():
            return
        try:
            # print(f"Info: Opening of communication from MFF101 SN {self.serial} in progress...")
            # print(f"Info: DeviceManagerCLI loading...")
            DeviceManagerCLI.BuildDeviceList()
            # print(f"Info: Connection with hard serial number...")
            self.device = FilterFlipper.CreateFilterFlipper(self.serial)
            self.device.Connect(self.serial)
            sleep(SETUP_INSTR["TIMECOM_S"])
            # print(f"Info: Hardware initialisation...")
            self._init_hardware()
            print(f"flipping mirror connected.")
        except Exception as error:
            self.disconnect()
            raise InstrumentNotFoundError(f"[open_communication] Communication failure with MFF101.")

    def disconnect(self) -> None:
        """ Fermeture de la communication avec l'instrument. """
        if not self.get_communication():
            return
        if isinstance(self.device, FilterFlipper):
            try:
                if self.is_polling:
                    self.stop_polling()
                self.device.Disconnect(True)
                print('Flipping mirror disconnected')
            except:
                print(f"Warning: Unable to properly close the MFF101.")
        self._init_communication_settings()
        self._init_settings()

    def _init_hardware(self) -> None:
        """
        Initialise l'instrument avec les paramètres par défaut.
        -----------------------------------------------------
        Raises:
            RemoteError: Si échec de communication.
        """
        try:
            self._on = True
            self.device.EnableDevice()  # Active l'électronique du moteur
            sleep(SETUP_INSTR["TIMECOM_S"])
            # print(f"Info: Dynamic hardware identification...")
            self.identify()
            sleep(SETUP_INSTR["TIMECOM_S"])
            # print(f"Info: Polling...")
            self.start_polling()
            sleep(SETUP_INSTR["TIMECOM_S"])
            # print(f"Info: Get initial position..")
            self.get_position()
        except Exception as error:
            raise type(error)(f"[_init_hardware] Hardware initialization failed: {error}")

    def identify(self) -> None:
        """
        Tentative d'identification materielle.
        ------------------------------------
        Returns:
            str: Identification du matériel.

        Raises:
            RemoteError: Si échec de communication.
            RuntimeWarning: Si communication fermée.
        """
        if not self.get_communication():
            raise RuntimeWarning(f"[identify] Communication to MFF101 is closed.")
        self.idn = self.device.DeviceID
        if self.idn is not None:
            # print(f"Info: Hardware identification success: {self.idn}.")
            pass
        else:
            raise RemoteError(f"[identify] Hardware identification failure.")

    def start_polling(self) -> None:
        """
        Interroge périodiquement le contrôlleur sur sa position.
        ---------------------------------------------------
        Notes:
            Utile en cas de fils manuel, indetectable autrement.

        Raises:
            RemoteError: Si échec de communication.
            RuntimeWarning: Si communication fermée ou polling déjà actif.
        """
        if not self.get_communication():
            raise RuntimeWarning(f"[start_polling] Communication to MFF101 is closed.")
        if self.is_polling:
            raise RuntimeWarning(f"[start_polling] Polling is already running.")
        try:
            self.device.StartPolling(self.polling_delay_ms)
            self.is_polling = True
            # print(f"Info: Polling successfully activated. Period of {self.polling_delay_ms}ms.")
        except Exception as error:
            raise RemoteError(f"[start_polling] Communication failure with MFF101: {error}.")

    def stop_polling(self) -> None:
        """
        Met fin à la mise à jour automatique de la position.
        --------------------------------------------------
        Raises:
            RemoteError: Si échec de communication.
            RuntimeWarning: Si communication fermée.
        """
        if not self.get_communication():
            raise RuntimeWarning(f"[stop_polling] Communication to MFF101 is closed.")
        if not self.is_polling:
            raise RuntimeWarning(f"[stop_polling] Polling is already closed.")
        try:
            self.device.StopPolling()
            self.is_polling = False
            # print(f"Info: Polling disabled.")
        except Exception as error:
            raise RemoteError(f"[stop_polling] Communication failure with MFF101: {error}.")

    def set_polling_delay(self, new_polling_delay: int) -> None:
        """
        Fixe un nouveau delais de vérification automatique de position.
        -----------------------------------------------------------
        Raises:
            RuntimeWarning: Si communication fermée.
            TypeError: Si polling_delay_ms non int.
            ValueError: Si polling_delay_ms hors limite.
            RemoteError: Si échec de communication.
        """
        if not self.get_communication():
            raise RuntimeWarning(f"[set_polling_delay] Communication to MFF101 is closed.")
        if not isinstance(new_polling_delay, int):
            try:
                new_polling_delay = int(new_polling_delay)
            except:
                raise TypeError(f"[set_polling_delay] Arg 'new_polling_delay' must be int and not '{type(new_polling_delay).__name__}'.")
        if not (self.polling_delay_min < new_polling_delay < self.polling_delay_max):
            raise ValueError(f"[set_polling_delay] Arg 'new_polling_delay' out of bounds. (new_polling_delay={new_polling_delay} | Min={self.polling_delay_min} | Max={self.polling_delay_max})")
        try:
            is_polling = self.is_polling
            if self.is_polling:
                self.stop_polling()
            self.polling_delay_ms = new_polling_delay
            if is_polling:
                self.start_polling()
        except Exception as error:
            raise type(error)(f"[set_polling_delay] {error}.") from error

    def get_polling_delay(self) -> int:
        """
        Retourne le delais automatique de verif de positionnement.
        -----------------------------------------------------
        Returns:
            int : Delais de vérification de positionnement.

        Raises:
            RuntimeWarning: Si communication fermée.
        """
        if not self.get_communication():
            raise RuntimeWarning(f"[get_polling_delay] Communication to MFF101 is closed.")
        return self.polling_delay_ms

    def get_position(self) -> str:
        """
        Récupère la position courrante du moteur.
        -------------------------------------
        Returns:
            str: "spatial" si position 1, "spectral" si position 2.

        Raises:
            RuntimeWarning: Si communication fermée.
            RemoteError: Si échec de communication
        """
        if not self.get_communication():
            raise RuntimeWarning(f"[get_position] Communication to MFF101 is closed.")
        try:
            if not self.is_polling:
                self.start_polling()
                sleep(self.polling_delay_ms/1000 + SETUP_INSTR["TIMECOM_S"])
                self.stop_polling()
            position = self.device.Position
            return "spatial" if position == 1 else "spectral"
        except Exception as error:
            raise type(error)(f"[get_position] {error}")

    def set_position_1(self, verbose: bool = False) -> None:
        """
        Oblige le MFF à passer en position 1 (vertical pour le bras spatial).
        --------------------------------------------------------------------------
        Raises:
            RuntimeWarning: Si communication fermée.
            TimeoutError: Si timeout atteint.
            RemoteError : Si échec de communication.
        """
        if not self.get_communication():
            raise RuntimeWarning(f"[get_position] Communication to MFF101 is closed.")
        try:
            self.target_position = 1
            self.ini_move_time = time()
            self.cur_move_time = 0
            self.device.SetPosition(0x01, 0)
            while not self.wait_for_target_position():
                sleep(SETUP_MFF["POSITION_DELAY_S"])
            if verbose:
                print(f"Flipping mirror set to the spatial arm.")
        except TimeoutError as error:
            raise type(error)(f"[set_position_1] {error}") from error
        except Exception as error:
            raise RemoteError(f"[set_position_1] Flip to position 1 failure: {error}.")

    def set_position_2(self, verbose: bool = False) -> None:
        """
        Oblige le MFF à passer en position 2 (horizontal pour le bras spectral).
        ---------------------------------------------------------------
        Raises:
            RuntimeWarning: Si communication fermée.
            TimeoutError: Si timeout atteint.
            RemoteError : Si échec de communication.
        """
        if not self.get_communication():
            raise RuntimeWarning(f"[get_position] Communication to MFF101 is closed.")
        try:
            self.target_position = 2
            self.ini_move_time = time()
            self.cur_move_time = 0
            self.device.SetPosition(0x02, 0)
            while not self.wait_for_target_position():
                sleep(SETUP_MFF["POSITION_DELAY_S"])
            if verbose:
                print(f"Flipping mirror set to the spectral arm.")
        except TimeoutError as error:
            raise type(error)(f"[set_position_2] {error}") from error
        except Exception as error:
            raise RemoteError(f"[set_position_2] Flip to position 2 failure: {error}.")

    def flip_position(self) -> None:
        """
        Permet de passer d'une position à une autre.
        --------------------------------------
        Raises:
            RuntimeWarning: Si communication fermée.
            TimeoutError: Si timeout atteint.
            RemoteError : Si échec de communication.
        """
        if not self.get_communication():
            raise RuntimeWarning(f"[switch_position] Communication to MFF101 is closed.")
        try:
            if not self.is_polling:
                self.start_polling()
            self.ini_move_time = time()
            self.set_position_1() if self.get_position() == "spectral" else self.set_position_2()
        except TimeoutError as error:
            raise type(error)(f"[flip_position] {error}") from error
        except Exception as error:
            raise RemoteError(f"[switch_position] Switch position failure: {error}.")

    def wait_for_target_position(self) -> bool:
        """
        Permet de s'assurer que la position cible est atteinte sans dépasser le timeout.
        ----------------------------------------------------------------------
        Returns:
            bool : True si position atteinte, False sinon.

        Raises:
            TimeoutError: Si timeout atteint.
        """
        self.cur_move_time = time() - self.ini_move_time
        if self.cur_move_time > SETUP_INSTR["TIMEOUT_S"]:
            raise TimeoutError(f"[wait_for_target_position] Timeout expired.")
        return self.device.Position == self.target_position

    def set_position(self, position: str, verbose: bool = False) -> None:
        """
        Change la position selon l'argument "spatial" ou "spectral".
        -----------------------------------------------------------
        Args:
            position (str): "spatial" ou "spectral"
        """
        current = self.get_position()
        if current == position:
            if verbose:
                print(f"Flipping mirror already in {position} position.")
            return
        if position == "spatial":
            self.set_position_1(verbose)
        elif position == "spectral":
            self.set_position_2(verbose)
        else:
            raise ValueError(f"[set_position] Invalid position: {position}. Use 'spatial' or 'spectral'.")