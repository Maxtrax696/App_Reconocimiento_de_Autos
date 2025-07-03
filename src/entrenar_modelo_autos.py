import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd

# Verificar GPU
print("GPU disponible:", tf.config.list_physical_devices('GPU'))

# Directorios
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset_dividido')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Hiperparámetros
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_NAME = 'modelo_autos.keras'

# Generadores de datos
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.1,
                               width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
val_test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(os.path.join(BASE_DIR, 'train'),
                                           target_size=IMG_SIZE,
                                           batch_size=BATCH_SIZE,
                                           class_mode='categorical')

val_data = val_test_gen.flow_from_directory(os.path.join(BASE_DIR, 'val'),
                                            target_size=IMG_SIZE,
                                            batch_size=BATCH_SIZE,
                                            class_mode='categorical')

test_data = val_test_gen.flow_from_directory(os.path.join(BASE_DIR, 'test'),
                                             target_size=IMG_SIZE,
                                             batch_size=BATCH_SIZE,
                                             class_mode='categorical',
                                             shuffle=False)

# Modelo base
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(train_data.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Compilación
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Checkpoint
checkpoint_path = os.path.join(MODEL_DIR, MODEL_NAME)
checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True,
                             monitor='val_accuracy', mode='max')

# Entrenamiento
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[checkpoint])

# Evaluación
loss, acc = model.evaluate(test_data)
print(f"Evaluación en TEST | Loss: {loss:.4f} | Accuracy: {acc:.4f}")
print(f"Modelo final guardado en: {checkpoint_path}")

# Guardar historial en CSV
history_df = pd.DataFrame(history.history)
history_csv_path = os.path.join(MODEL_DIR, "historial_entrenamiento.csv")
history_df.to_csv(history_csv_path, index=False)
print(f"Historial de entrenamiento guardado en: {history_csv_path}")

# Convertir tensores para guardar en JSON
def convertir_tensores(obj):
    if isinstance(obj, dict):
        return {k: convertir_tensores(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convertir_tensores(v) for v in obj]
    elif hasattr(obj, 'numpy'):
        return obj.numpy().tolist()
    return obj

history_json_path = os.path.join(MODEL_DIR, "historial_entrenamiento.json")
with open(history_json_path, 'w') as f:
    json.dump(convertir_tensores(history.history), f, indent=4)
print(f"Historial JSON guardado en: {history_json_path}")
