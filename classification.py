import numpy as np
from tqdm import tqdm
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# -------------------------------------------------------------------------
# Fonction : extract_hog_features
# Objectif  : Extraire les features HOG de chaque image
# -------------------------------------------------------------------------
data = np.load("dataset.npz", allow_pickle=True)
X_train = data["X_train"]
X_val   = data["X_val"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_val   = data["y_val"]
y_test  = data["y_test"]



def extract_hog_features(image_paths, desc="HOG"):
    """Extrait les features HOG de chaque image."""
    features = []
    for path in tqdm(image_paths, desc=desc):
        try:
            img = Image.open(path).convert("RGB")  # Ouvre et convertit en RGB
            img = np.array(img)                    # Convertit en array (uint8 0-255)
            feat = hog(
                img,
                orientations=12,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                channel_axis=-1
            )
            features.append(feat)
        except Exception as e:
            print(f"Erreur sur {path}: {e}")
    return np.array(features)

print("Extraction des features HOG (train)...")
X_train_hog = extract_hog_features(X_train, "Train HOG")

print("Extraction des features HOG (val)...")
X_val_hog = extract_hog_features(X_val, "Val HOG")

print("Extraction des features HOG (test)...")
X_test_hog = extract_hog_features(X_test, "Test HOG")

print(f"\nShape features HOG : {X_train_hog.shape}")

print("=" * 50)
print("Modèle A — Random Forest (Baseline)")
print("=" * 50)

t0 = time.time()

rf = RandomForestClassifier(
    n_estimators=100,    # nombre d'arbres
    max_depth=50,        # profondeur maximale des arbres
    n_jobs=-1,           # utilise tous les cœurs CPU
    random_state=42
)

# Entraînement
rf.fit(X_train_hog, y_train)
t1 = time.time()

# Prédiction sur la validation
y_pred_rf = rf.predict(X_val_hog)
acc_rf = accuracy_score(y_val, y_pred_rf)

print(f"  Temps d'entraînement : {t1 - t0:.1f}s")
print(f"  Accuracy (validation) : {acc_rf * 100:.1f}%")