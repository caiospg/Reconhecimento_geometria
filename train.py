# train_final.py
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = (128,128)
BATCH_SIZE = 64
EPOCHS = 30
DATA_DIR = Path("data")
MODEL_PATH = "mobilenet_shapes_final.keras"

# --- Datasets ---
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR/"train", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int', shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR/"val", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR/"test", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int', shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Ordem de classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# --- Data augmentation leve ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# --- MobileNetV2 pré-treinado ---
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE+(3,), include_top=False, weights="imagenet"
)
base_model.trainable = False

inputs = layers.Input(shape=IMG_SIZE+(3,))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])
model.summary()

# --- Callbacks ---
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

# --- Treino inicial ---
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# --- Fine-tuning ---
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

history_ft = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

# --- Avaliação ---
loss, acc = model.evaluate(test_ds)
print(f"✅ Test accuracy final: {acc*100:.2f}%")
