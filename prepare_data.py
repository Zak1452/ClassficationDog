import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# -------------------------------------------------------------------------
# Author : CHAKER Zakaria & NSANGUE Nathan
# Date   : 16/03/2026
# Version: 1.0
#
# Objectif :
# Préparer les données à partir du dossier "resized/".
#
# Cette étape permet de :
# - Lire toutes les images redimensionnées
# - Associer chaque image à sa classe (race)
# - Encoder les labels (texte en entiers)
# - Mélanger les données
# - Decoupage en train/validation/test
#
# Sortie :
# - X_train, X_val, X_test : chemins des images
# - y_train, y_val, y_test : labels encodés
# -------------------------------------------------------------------------

BASE_DIR = "./stanford_dogs"
RESIZED_DIR = os.path.join(BASE_DIR, "resized")

# Vérification
if not os.path.exists(RESIZED_DIR):
    print("Le dossier resized est introuvable. Lance d'abord le script de redimensionnement.")
    exit(1)

# Récupération des classes
breed_folders = sorted([
    d for d in os.listdir(RESIZED_DIR)
    if os.path.isdir(os.path.join(RESIZED_DIR, d))
])

# Création des données (chemins + labels)
image_paths = []
labels = []

for folder in breed_folders:
    path = os.path.join(RESIZED_DIR, folder)
    imgs = glob.glob(os.path.join(path, "*.jpg"))
    
    for img_path in imgs:
        image_paths.append(img_path)
        labels.append(folder.split("-", 1)[-1])

image_paths = np.array(image_paths)
labels = np.array(labels)

print(f"\nNombre total d'images : {len(image_paths)}")

# Encodage des labels
le = LabelEncoder() #Encodage des labels avec Classe LabelEncoder
y_encoded = le.fit_transform(labels)

# Mélange des données
image_paths, y_encoded = shuffle(image_paths, y_encoded, random_state=42)


# Decoupage 70% train / 15% validation / 15% test
X_train, X_temp, y_train, y_temp = train_test_split(image_paths, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# Affichage
print("\nRepartition des donnees :")
print(f"Train : {len(X_train)} images ({len(X_train)/len(image_paths)*100:.0f}%)")
print(f"Validation : {len(X_val)} images ({len(X_val)/len(image_paths)*100:.0f}%)")
print(f"Test : {len(X_test)} images ({len(X_test)/len(image_paths)*100:.0f}%)")