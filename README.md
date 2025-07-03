# Car Recognition AI

Este proyecto implementa una IA que reconoce autos a partir de imágenes y los clasifica por marca, modelo y año utilizando TensorFlow + Keras.

## Estructura
- `dataset_dividido/`: Contiene las imágenes divididas en `train/`, `val/` y `test/`
- `src/`: Scripts de entrenamiento, predicción y organización del dataset
- `models/`: Modelos entrenados (.keras)

car-recognition-ai/
│
├── dataset_dividido/           # ⚠️ NO se sube a GitHub (muy pesado, usar .gitignore)
│   ├── train/
│   ├── val/
│   └── test/
│
├── notebooks/                  # Análisis y experimentos en Jupyter
│   └── exploracion_dataset.ipynb
│
├── src/                        # Código fuente del proyecto
│   ├── __init__.py
│   ├── reorganizar_y_dividir_dataset.py
│   ├── entrenar_modelo_autos.py
│   └── predecir_imagen.py         # (opcional) para usar el modelo en nuevas imágenes
│
├── models/                     # Modelos entrenados (.keras o .h5)
│   └── modelo_autos.keras
│
├── scripts/                    # (opcional) shell scripts o comandos automatizados
│   └── convertir_dataset.sh
│
├── requirements.txt            # Dependencias del proyecto
├── .gitignore                  # Archivos/carpetas que no deben subirse
├── README.md                   # Explicación del proyecto
└── LICENSE                     # (opcional) Licencia abierta como MIT, GPL, etc.


## Entrenamiento
```bash
python src/entrenar_modelo_autos.py
```

## Uso de modelo entrenado
```bash
python src/predecir_imagen.py --image test.jpg
```

## Requisitos
instalar con:
```bash
pip install -r requirements.txt
```

## Licencia
MIT = permiso de uso: modificar, distribuir el proyecto incluso comercialmente, con atribucion
