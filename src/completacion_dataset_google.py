import os
from PIL import Image
from icrawler.builtin import GoogleImageCrawler

# Configuración
CARPETA_TRAIN = "dataset_dividido/train"
MIN_IMAGENES = 500
MIN_RESOLUCION = (150, 150)
EXTENSIONES_VALIDAS = ('.jpg', '.jpeg', '.png')

def contar_imagenes_validas(carpeta):
    return len([
        f for f in os.listdir(carpeta)
        if f.lower().endswith(EXTENSIONES_VALIDAS)
    ])

def eliminar_imagenes_invalidas(carpeta):
    eliminadas = 0
    for nombre in os.listdir(carpeta):
        ruta = os.path.join(carpeta, nombre)
        if not nombre.lower().endswith(EXTENSIONES_VALIDAS):
            os.remove(ruta)
            eliminadas += 1
            continue
        try:
            with Image.open(ruta) as img:
                if img.size[0] < MIN_RESOLUCION[0] or img.size[1] < MIN_RESOLUCION[1]:
                    os.remove(ruta)
                    eliminadas += 1
        except Exception:
            os.remove(ruta)
            eliminadas += 1
    return eliminadas

def descargar_imagenes(clase, destino, cantidad):
    print(f"🌐 Descargando {cantidad} imágenes para: {clase}")
    crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=2,
        downloader_threads=4,
        storage={"root_dir": destino}
    )
    # Filtra solo archivos JPEG o PNG y evita SVG, ICO, etc.
    crawler.crawl(
        keyword=clase,
        max_num=cantidad,
        file_idx_offset=len(os.listdir(destino)),
        filters={"type": "photo", "size": "medium"}
    )

def completar_dataset():
    clases = sorted(next(os.walk(CARPETA_TRAIN))[1])
    print(f"📂 Clases detectadas: {len(clases)}\n")

    for clase in clases:
        clase_texto = clase.replace("_", " ")
        carpeta_clase = os.path.join(CARPETA_TRAIN, clase)

        eliminadas = eliminar_imagenes_invalidas(carpeta_clase)
        if eliminadas:
            print(f"🧹 {eliminadas} imágenes inválidas eliminadas de: {clase_texto}")

        actuales = contar_imagenes_validas(carpeta_clase)

        if actuales < MIN_IMAGENES:
            faltan = MIN_IMAGENES - actuales
            print(f"⚠️  {clase_texto:<30} → tiene {actuales} imágenes → faltan {faltan}")
            descargar_imagenes(clase_texto, carpeta_clase, faltan)

            eliminadas = eliminar_imagenes_invalidas(carpeta_clase)
            if eliminadas:
                print(f"🧹 {eliminadas} imágenes inválidas eliminadas después de la descarga")
        else:
            print(f"✅ {clase_texto:<30} → OK con {actuales} imágenes")

    # Resumen
    print("\n📊 Resumen final:")
    for clase in clases:
        ruta = os.path.join(CARPETA_TRAIN, clase)
        total = contar_imagenes_validas(ruta)
        print(f"📁 {clase:<30} → {total} imágenes válidas")

    print("\n✅ Dataset enriquecido y limpio 🎯")

if __name__ == "__main__":
    completar_dataset()
