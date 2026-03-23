import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# -------------------------------------------------------------------------
# Author : CHAKER Zakaria & NSANGUE Nathan
# Date   :16/03/2026
# Version: 1.2
#
# Objectif :
# Redimensionnement des images.
#
# Cette étape permet de :
# - Parcourir l’ensemble des classes (races de chiens)
# - Limiter le nombre d’images par classe (MAX_PER_CLASS)
# - Redimensionner toutes les images à une taille uniforme (128 x 128)
# - Sauvegarder les images prétraitees au format JPG dans un nouveau dossier (resized/)
#
# Sortie :
# - Dossier "resized/" contenant les images redimensionnées -> organisées par race (même structure que le dataset original)
# -------------------------------------------------------------------------

IMG_SIZE = 128 # Taille cible (128x128 pixels)
MAX_PER_CLASS = 140  # ombre maxi des echantillons à redimensioner

BASE_DIR = "./stanford_dogs"
IMAGES_DIR = os.path.join(BASE_DIR, "Images")

# Vérification de l'existence du dossier Images /*Prerequis*/
if not os.path.exists(IMAGES_DIR):
    print("Le dossier Images est introuvable. Vérifiez que le dataset est bien téléchargé.")
    exit(1)     

RESIZED_DIR = os.path.join(BASE_DIR, "resized")
os.makedirs(RESIZED_DIR, exist_ok=True)

try:
    breed_folders = sorted([
        d for d in os.listdir(IMAGES_DIR)
        if os.path.isdir(os.path.join(IMAGES_DIR, d))
    ])
except Exception as e:
    print("Erreur lors de la lecture du dossier Images :", e)
    exit(1)              


for folder in tqdm(breed_folders, desc="Traitement par race"):
    path = os.path.join(IMAGES_DIR, folder)
    imgs = glob.glob(os.path.join(path, "*.jpg"))
    
    # Limite MAX_PER_CLASS
    if MAX_PER_CLASS is not None:
        imgs = imgs[:MAX_PER_CLASS]
    
    # Créer dessous-dossier pour enregistrer les images redimensionnées
    save_dir = os.path.join(RESIZED_DIR, folder)
    os.makedirs(save_dir, exist_ok=True)
    
    for img_path in imgs:
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            
            # Nouveau nom de fichier
            save_path = os.path.join(save_dir, os.path.basename(img_path))
            
            # Savig en JPG
            img.save(save_path, format="JPEG", quality=95)
            
        except Exception as e:
            print(f"Erreur sur {img_path} : {e}")