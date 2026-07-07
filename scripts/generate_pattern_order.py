# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 15:23:28 2025

@author: equipe-onli
"""

import numpy as np
import os
import shutil
os.chdir('E:\\openspyrit\\spas\\scripts')
################ input #######################
Np = 128
pattern_thickness = 4
############## begin ##########################
pattern_dim = '1D'
scan_mode = 'Walsh'
pattern_order_source_bu = '../stats/' + pattern_dim + '/BU/pattern_order_' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np) + '.npz'
pattern_order_source = '../stats/' + pattern_dim + '/pattern_order_' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np) + '.npz'

pattern_order=np.arange(Np, dtype=np.uint16)

np.savez(pattern_order_source[:len(pattern_order_source)-4], pattern_order = pattern_order, pos_neg = False)

############# read pattern order #############
a=np.load(pattern_order_source)
print(a['pattern_order'])

#%% rename pattern by Np
import os
import shutil
Npo = 256
pattern_origine_source = '../Patterns/' + pattern_dim + '/' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Npo)
Np = 128
pattern_source = '../Patterns/' + pattern_dim + '/' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np)
if not os.path.exists(pattern_source):
    os.makedirs(pattern_source)

for i in range(Np):
    pattern_origin_name = pattern_origine_source + '/' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Npo) + '_' + str(i) + '.png'
    pattern_name = pattern_source + '/' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np) + '_' + str(i) + '.png'
    shutil.copyfile(pattern_origin_name, pattern_name)
#%% rename pattern by scan_mode
Np = 128
pattern_thickness = 4
pattern_dim = '1D'
scan_mode_origin = 'Walsh'
pattern_origine_source = '../Patterns/' + pattern_dim + '/' + scan_mode_origin + '_' + str(pattern_thickness) + 'x' + str(Np)
scan_mode = 'Walsh_sparse'
pattern_source = '../Patterns/' + pattern_dim + '/' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np)
if not os.path.exists(pattern_source):
    os.makedirs(pattern_source)

for i in range(Np):
    pattern_origin_name = pattern_origine_source + '/' + scan_mode_origin + '_' + str(pattern_thickness) + 'x' + str(Np) + '_' + str(i) + '.png'
    pattern_name = pattern_source + '/' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np) + '_' + str(i) + '.png'
    shutil.copyfile(pattern_origin_name, pattern_name)    
#%% invert pattern
import imageio.v3 as iio
import numpy as np
from PIL import Image
################ input #######################
Np = 512
pattern_thickness = 16
pattern_dim = '1D'
scan_mode = 'Walsh'
fold = "../Patterns/1D/Walsh_" + str(pattern_thickness) + "x" + str(Np)
if os.path.isdir(fold) == False:
    os.mkdir(fold)
else:
    print("folder already exist")
############## begin ##########################

pattern_origine_source_bu = '../Patterns/' + pattern_dim + '/BU/' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np)
for i in range(Np):
# i = 5
    pattern_origin_name = pattern_origine_source_bu + '/' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np) + '_' + str(i) + '.png'
    pattern = iio.imread(pattern_origin_name)
    # pattern_temp = np.empty((pattern.shape), dtype=np.int16)
    pattern_temp = pattern
    pattern_temp2 = pattern_temp.astype(np.int16)
    pattern_temp2 = abs(pattern_temp2 - 255)
    pattern_temp3 = pattern_temp2.astype(np.uint8)
    
    pattern_source = '../Patterns/' + pattern_dim + '/' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np)
    pattern_name = pattern_source + '/' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np) + '_' + str(i) + '.png'
    
    im = Image.fromarray(pattern_temp3)
    im.save(pattern_name)
#%% fill the sparse pattern
import imageio.v3 as iio
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
################ input #######################
Np = 512
pattern_thickness = 16
pattern_dim = '1D'
scan_mode = 'Walsh'
############## def ############################
def remplir_pixels_4voisins(image):
    # On crée une copie pour ne pas modifier l'image originale
    result = image.copy()
    h, w = image.shape

    for i in range(1, h-1):
        for j in range(1, w-1):
            if image[i, j] == 0:
                # Vérifier les 4 voisins directs
                if (image[i-1, j] == 255 and
                    image[i+1, j] == 255 and
                    image[i, j-1] == 255 and
                    image[i, j+1] == 255):
                    result[i, j] = 255
    return result
############## begin ##########################
pattern_source = '../Patterns/' + pattern_dim + '/' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np)
for i in range(Np):
    pattern_name = pattern_source + '/' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np) + '_' + str(i) + '.png'
    pattern = iio.imread(pattern_name)
    
    pattern_filled = remplir_pixels_4voisins(pattern)
    im = Image.fromarray(pattern_filled)
    im.save(pattern_name)

# plt.figure()
# plt.imshow(pattern)

# plt.figure()
# plt.imshow(pattern_filled)
#%% turn the patterns at 45°
import imageio.v3 as iio
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
################ input #######################
Np = 128
pattern_thickness = 4
pattern_dim = '1D'
scan_mode = 'Walsh'
############## begin ##########################
pattern_source = '../Patterns/' + pattern_dim + '/' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np)
pattern_dim_new = '1D_45'
pattern_source_new = '../Patterns/' + pattern_dim_new + '/' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np)
heigh_dmd = 1024
width_dmd = 768
x = int(heigh_dmd / 4)
y = int(width_dmd / 4)
lx = x + int(heigh_dmd / 2)
ly = y + int(width_dmd / 2)

for i in range(Np):
    pattern_name = pattern_source + '/' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np) + '_' + str(i) + '.png'
    pattern_name_new = pattern_source_new + '/' + scan_mode + '_' + str(pattern_thickness) + 'x' + str(Np) + '_' + str(i) + '.png'
    img = Image.open(pattern_name)
    # plt.figure()
    # plt.imshow(img)
    # plt.title('original image')
    
    sub_image = img.crop(box=(x,y,lx,ly)).rotate(45)
    pixels = sub_image.load()
    colonne1 = 253 #258
    colonne2 = 258
    for yi in range(sub_image.size[1]):
        pixels[colonne1, yi] = 0
        pixels[colonne2, yi] = 0

    img.paste(sub_image, box=(x,y))

    img.save(pattern_name_new)

    # plt.figure()
    # plt.imshow(img)
    # plt.title('image tilted')
#%%
import numpy as np
from PIL import Image

def hadamard_1d(width, length=384):
    """
    Génère un motif de Hadamard 1D de taille (length x width) et l'exporte en PNG.

    Args:
        width (int): Largeur du motif en pixels (4 ou 8).
        length (int): Longueur du motif en pixels (par défaut 384).
    """
    # Vérification des paramètres
    if width not in [4, 8]:
        raise ValueError("La largeur doit être 4 ou 8 pixels.")

    # Génération de la matrice de Hadamard 1D
    hadamard_order = int(np.log2(width))
    hadamard_matrix = np.array([1])

    for _ in range(hadamard_order):
        hadamard_matrix = np.vstack([
            np.hstack([hadamard_matrix, hadamard_matrix]),
            np.hstack([hadamard_matrix, -hadamard_matrix])
        ])

    # Répétition pour atteindre la longueur souhaitée
    hadamard_pattern = np.tile(hadamard_matrix[-1], length // width + 1)[:length]

    # Normalisation pour l'affichage (0 = noir, 255 = blanc)
    hadamard_pattern = (hadamard_pattern + 1) * 127.5
    hadamard_pattern = hadamard_pattern.astype(np.uint8)

    # Création de l'image
    image_array = np.tile(hadamard_pattern, (width, 1))
    image = Image.fromarray(image_array, mode='L')

    # Sauvegarde en PNG
    image.save(f"hadamard_1d_width_{width}_length_{length}.png")
    print(f"Motif de Hadamard 1D sauvegardé : hadamard_1d_width_{width}_length_{length}.png")

# Exemple d'utilisation
hadamard_1d(width=4, length=384)
# hadamard_1d(width=8, length=384)


















