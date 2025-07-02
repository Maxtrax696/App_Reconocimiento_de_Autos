# Car Recognition AI

Este proyecto implementa una IA que reconoce autos a partir de imágenes y los clasifica por marca, modelo y año utilizando TensorFlow + Keras.

## Estructura
- `dataset_dividido/`: Contiene las imágenes divididas en `train/`, `val/` y `test/`
- `src/`: Scripts de entrenamiento, predicción y organización del dataset
- `models/`: Modelos entrenados (.keras)

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
MIT
