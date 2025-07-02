import os
import shutil
import random

# Configura aquí tu carpeta de origen con TODAS las imágenes
ORIGEN = r"D:\UCE\8VO SEMESTRE\MINERIA DE DATOS\PROPUESTA_PROYECTO\PROYECTO_FINAL\DATASET_CARS"
DESTINO = "dataset_dividido"
SUBSETS = ["train", "val", "test"]

# Porcentajes de división
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

assert abs((TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT) - 1.0) < 0.01, "Los porcentajes deben sumar 100%"

# Crear estructura base
for subset in SUBSETS:
    os.makedirs(os.path.join(DESTINO, subset), exist_ok=True)

# Recolectar y agrupar imágenes por clase (Marca_Modelo_Año)
clases = {}

for nombre_archivo in os.listdir(ORIGEN):
    if not nombre_archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    partes = nombre_archivo.split("_")
    if len(partes) < 3:
        continue

    clase = "_".join(partes[:3])  # Ej: "Toyota_Corolla_2020"
    clases.setdefault(clase, []).append(nombre_archivo)

# Procesar cada clase
for clase, archivos in clases.items():
    print(f"Clase '{clase}': {len(archivos)} imágenes")

    random.shuffle(archivos)

    n_total = len(archivos)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val = int(n_total * VAL_SPLIT)
    n_test = n_total - n_train - n_val

    splits = {
        "train": archivos[:n_train],
        "val": archivos[n_train:n_train + n_val],
        "test": archivos[n_train + n_val:]
    }

    for subset, subset_files in splits.items():
        subset_dir = os.path.join(DESTINO, subset, clase)
        os.makedirs(subset_dir, exist_ok=True)

        for file_name in subset_files:
            origen_path = os.path.join(ORIGEN, file_name)
            destino_path = os.path.join(subset_dir, file_name)
            shutil.copy2(origen_path, destino_path)

print("Reorganización y división completadas.")
