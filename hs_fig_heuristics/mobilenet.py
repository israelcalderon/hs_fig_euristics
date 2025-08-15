import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from typing import Any
import numpy as np


def create_model(num_classes: int, X_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray)-> tuple[Model, Any]:
    y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
    y_val_one_hot = to_categorical(y_val, num_classes=num_classes)

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32 # Cuántas imágenes procesar a la vez

    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,),
                             include_top=False,
                             weights='imagenet')
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    print("Compilando el modelo...")
    model.compile(optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()
    train_generator = datagen.flow(X_train, y_train_one_hot, batch_size=BATCH_SIZE)
    print("Iniciando el entrenamiento...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=(x_val, y_val_one_hot), # Pasamos los datos de validación directamente
        epochs=10
    )
    return model, history
