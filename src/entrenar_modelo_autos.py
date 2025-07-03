import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import os

# Verificar GPU
print("GPU disponible:", tf.config.list_physical_devices('GPU'))

# Ruta al dataset organizado
BASE_DIR = "dataset_dividido"

# Parámetros
IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 50

# Generadores de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Modelo base preentrenado
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
base_model.trainable = False

# Agregar capas personalizadas
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compilar modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callback EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenamiento
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# Evaluación
loss, accuracy = model.evaluate(test_generator)
print(f"\nEvaluación en test - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Guardar modelo
model.save("modelo_autos.h5")
print("Modelo guardado como 'modelo_autos.h5'")

# Guardar historial como CSV
hist_df = pd.DataFrame(history.history)
hist_df.to_csv("historial_entrenamiento.csv", index=False)
print("Historial guardado como 'historial_entrenamiento.csv'")
