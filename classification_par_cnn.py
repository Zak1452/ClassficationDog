import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


data = np.load("dataset.npz", allow_pickle=True)

X_train = data["X_train"]
X_val   = data["X_val"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_val   = data["y_val"]
y_test  = data["y_test"]


def load_images(paths, name="dataset"):
    images = []
    total = len(paths)

    print(f"\nChargement des images ({name})...")
    print(f"Total : {total} images")

    for i, p in enumerate(tqdm(paths, desc=f"Loading {name}")):
        try:
            img = Image.open(p).convert("RGB")
            images.append(np.array(img))
        except Exception as e:
            print(f"\nErreur image : {p}")

        # Affichage toutes les 1000 images
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"{i+1}/{total} images chargees ({(i+1)/total*100:.1f}%)")

    images = np.array(images, dtype=np.float32) / 255.0

    print(f"\n{name} termine !")
    print(f"Shape : {images.shape}")
    print(f"Memoire : {images.nbytes / 1e6:.0f} MB")

    return images

print("Chargement images...")
X_train = load_images(X_train)
X_val   = load_images(X_val)
X_test  = load_images(X_test)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(len(np.unique(y_train)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit( X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, verbose=1)

print("\nEvaluation sur le test...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nAccuracy test : {test_acc*100:.2f}%")

# -------------------------------------------------------------------------
# Sauvegarde du modèle
# -------------------------------------------------------------------------
model_save_path = "cnn_dogs_model.h5"
model.save(model_save_path)
print(f"\nModèle sauvegardé dans : {model_save_path}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'],     label='Train',      color='steelblue', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation', color='coral',     linewidth=2)
axes[0].set_title('Accuracy par époque', fontsize=13)
axes[0].set_xlabel('Époque')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train',      color='steelblue', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation', color='coral',     linewidth=2)
axes[1].set_title('Loss par époque', fontsize=13)
axes[1].set_xlabel('Époque')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Courbes d'apprentissage — CNN", fontsize=14)
plt.tight_layout()
plt.savefig("courbes_apprentissage.png", dpi=150, bbox_inches='tight')
plt.show()
print("Courbes sauvegardées : courbes_apprentissage.png")
