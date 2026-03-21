import os
import urllib.request
import tarfile
import subprocess
import sys

# Author CHAKER Zakaria & NSANGUE Nathan
# Dans cette section nous récupérons le dataset présent en ligne...
# Ce script permet de télécharger et d'extraire le dataset en utilisant des bibliothèques adaptées
# comme tarfile pour l'extraction.
# Une vérification de la connexion Internet est effectuée avant le téléchargement.
# Gestion des erreurs ajoutée pour améliorer la robustesse.
# Version 1.1

BASE_DIR = "./stanford_dogs"
os.makedirs(BASE_DIR, exist_ok=True)

# Chargement des URL dans des variables 
IMAGES_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
ANNOTS_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
LISTS_URL  = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"

def check_internet():
    """
    Vérifie la connexion Internet en envoyant un ping vers 8.8.8.8.
    Retourne True si Internet est accessible, sinon False.
    """
    try:
        param = "-n" if sys.platform.lower() == "win32" else "-c"
        result = subprocess.run(
            ["ping", param, "1", "8.8.8.8"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return result.returncode == 0
    except Exception as e:
        print("Erreur lors de la vérification de la connexion Internet :", e)
        return False

def download_and_extract(url, dest_dir):
    """
    Télécharge et décompresse un fichier .tar
    """
    filename = os.path.join(dest_dir, url.split("/")[-1])

    # Téléchargement
    if not os.path.exists(filename):
        print(f"Téléchargement de {url.split('/')[-1]}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print("Téléchargement terminé.")
        except Exception as e:
            print("Erreur lors du téléchargement :", e)
            return
    else:
        print(f"{url.split('/')[-1]} déjà présent, on passe.")

    # Extraction
    print("Extraction en cours...")
    try:
        with tarfile.open(filename) as tar:
            tar.extractall(dest_dir)
        print("Extraction terminée.")
    except Exception as e:
        print("Erreur lors de l'extraction :", e)

# Vérification de la connexion Internet
if not check_internet():
    print("Aucune connexion Internet détectée. Veuillez vérifier votre réseau.")
    sys.exit(1)

# Télécharger desimages etlistes de train/test
download_and_extract(IMAGES_URL, BASE_DIR)
download_and_extract(LISTS_URL, BASE_DIR)

print("\nDataset prêt.")