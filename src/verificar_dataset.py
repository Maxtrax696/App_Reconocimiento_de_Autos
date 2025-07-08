import os

# Configuración
BASE_DIR = "dataset_dividido"
SUBSETS = ["train", "val", "test"]
EXTS = ('.jpg', '.png', '.jpeg')

def contar_imagenes_por_clase(subset_dir):
    conteo = {}
    clases = sorted(next(os.walk(subset_dir))[1])
    for clase in clases:
        clase_path = os.path.join(subset_dir, clase)
        num_imagenes = len([
            f for f in os.listdir(clase_path)
            if os.path.isfile(os.path.join(clase_path, f)) and f.lower().endswith(EXTS)
        ])
        conteo[clase] = num_imagenes
    return conteo

def mostrar_resumen(nombre, conteo):
    print(f"\n📂 Conjunto: {nombre.upper()}")
    print(f"🧾 Total clases: {len(conteo)}")
    total_imgs = sum(conteo.values())
    promedio = total_imgs / len(conteo) if conteo else 0
    print(f"📸 Total imágenes: {total_imgs}")
    print(f"📊 Promedio por clase: {promedio:.2f}\n")

    # Clases con menos de 30 imágenes
    for clase, cantidad in sorted(conteo.items(), key=lambda x: x[1]):
        if cantidad < 30:
            print(f"⚠️  Clase: {clase:<30} → {cantidad} imágenes")

# Procesar cada conjunto
for subset in SUBSETS:
    subset_dir = os.path.join(BASE_DIR, subset)
    if os.path.exists(subset_dir):
        conteo = contar_imagenes_por_clase(subset_dir)
        mostrar_resumen(subset, conteo)
    else:
        print(f"❌ Carpeta no encontrada: {subset_dir}")
