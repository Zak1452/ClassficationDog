import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# -------------------------------------------------------------------------
# Author CHAKER Zakaria & NSANGUE Nathan
# Objectif :
# Analyse exploratoire du dataset Stanford Dogs....
#
# Cette partie du code permet de :
# - Vérifier l'existence du dossier contenant les images.
# - Identifier les classes (races de chiens) présentes dans le dataset.
# - Compter le nombre d'images par classe.
# - Générer des statistiques globales (total, moyenne, min, max) relatives au Dataset
# - Analyser les dimensions des images (largeur et hauteur)
#
# Sorties :
# - distribution_races.png : répartition des images par race
# - dimensions_images.png : distribution des tailles d’images
#
# Remarque :
# Cette étape est essentielle pour comprendre la structure du dataset avant l’entraînement du modèle (équilibrage, taille des images, etc.).
# Date 15/03/2026
# Version 1.1
# -------------------------------------------------------------------------

BASE_DIR = "./stanford_dogs"
IMAGES_DIR = os.path.join(BASE_DIR, "Images")

# Vérification de l'existence du dossier Images
if not os.path.exists(IMAGES_DIR):
    print("Le dossier Images est introuvable. Vérifiez que le dataset est bien téléchargé.")
    exit(1)

# Lister toutes les races (= sous-dossiers)
try:
    breed_folders = sorted([
        d for d in os.listdir(IMAGES_DIR)
        if os.path.isdir(os.path.join(IMAGES_DIR, d))
    ])
except Exception as e:
    print("Erreur lors de la lecture du dossier Images :", e)
    exit(1)

# Affichage des résultats
print(f"Nombre de races : {len(breed_folders)}")

breed_counts = {}
for folder in breed_folders:
    path = os.path.join(IMAGES_DIR, folder)
    images = glob.glob(os.path.join(path, "*.jpg"))
    name = folder.split("-", 1)[-1].replace("_", " ").title()
    breed_counts[name] = len(images)

df_counts = pd.DataFrame.from_dict(breed_counts, orient="index", columns=["nb_images"])
df_counts = df_counts.sort_values("nb_images", ascending=False)

print(" Statistiques globales :")
print(f"  Total images     : {df_counts['nb_images'].sum()}")
print(f"  Races            : {len(df_counts)}")
print(f"  Moy. par race    : {df_counts['nb_images'].mean():.1f}")
print(f"  Min. par race    : {df_counts['nb_images'].min()}")
print(f"  Max. par race    : {df_counts['nb_images'].max()}")

# Visualisation : distribution du nombre d'images par race
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Top 20 races les plus représentées
top20 = df_counts.head(20)
axes[0].barh(top20.index[::-1], top20["nb_images"][::-1], color="steelblue")
axes[0].set_title("Top 20 races les plus représentées", fontsize=13)
axes[0].set_xlabel("Nombre d'images")
axes[0].axvline(df_counts["nb_images"].mean(), color="red", linestyle="--", label="Moyenne")
axes[0].legend()

# Distribution globale (affichage en histogramme)
axes[1].hist(df_counts["nb_images"], bins=20, color="coral", edgecolor="white")
axes[1].set_title("Distribution du nb d'images par race", fontsize=13)
axes[1].set_xlabel("Nombre d'images")
axes[1].set_ylabel("Nombre de races")
axes[1].axvline(df_counts["nb_images"].mean(), color="red", linestyle="--", label="Moyenne")
axes[1].legend()

plt.tight_layout()
plt.savefig("distribution_races.png", dpi=150, bbox_inches="tight")
plt.show()
print("Graphique sauvegarde : distribution_races.png")

print("Analyse des dimensions (echantillon 500 images)...")
widths, heights = [], []

all_images = glob.glob(os.path.join(IMAGES_DIR, "**", "*.jpg"), recursive=True)
sample_imgs = np.random.choice(all_images, min(500, len(all_images)), replace=False)

for img_path in sample_imgs:
    try:
        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
    except:
        pass

print(f"\nDimensions des images :")
print(f"  Largeur  — min: {min(widths)}, max: {max(widths)}, moy: {np.mean(widths):.0f}")
print(f"  Hauteur  — min: {min(heights)}, max: {max(heights)}, moy: {np.mean(heights):.0f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(widths, bins=30, color="teal", edgecolor="white")
axes[0].set_title("Distribution des largeurs")
axes[0].set_xlabel("Pixels")
axes[1].hist(heights, bins=30, color="orange", edgecolor="white")
axes[1].set_title("Distribution des hauteurs")
axes[1].set_xlabel("Pixels")
plt.tight_layout()
plt.savefig("dimensions_images.png", dpi=150, bbox_inches="tight")
plt.show()