"""
Training script for Skin Disease Detection using MobileNetV2/MobileNetV3 transfer learning.
Saves trained model to model/skin_disease_model.h5
"""

import os
import argparse
import json
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def build_model(input_shape=(224,224,3), num_classes=5, backbone='mobilenetv2', fine_tune_at=100):
    if backbone=='mobilenetv3':
        base = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = True
    # Freeze until the fine_tune_at layer
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base.input, outputs=outputs)
    return model


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to dataset folder')
    parser.add_argument('--backbone', type=str, default='mobilenetv2', choices=['mobilenetv2','mobilenetv3'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--save_to', type=str, default='model/skin_disease_model.h5')
    args = parser.parse_args()

    # Allow datasets placed inside `model/` (some users unpack datasets there)
    base_dir = args.data_dir
    if not os.path.exists(os.path.join(base_dir, 'train')):
        # check model/ for train/test folders
        candidate = os.path.join(os.path.dirname(__file__), 'model')
        if os.path.exists(os.path.join(candidate, 'train')):
            base_dir = candidate

    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    IMG_SIZE = (224,224)

    # Augmentation: if no explicit validation folder, use validation_split
    has_validation_folder = os.path.exists(val_dir)

    if has_validation_folder:
        train_aug = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.05,
            zoom_range=0.15,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest'
        )
        val_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_gen = train_aug.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=args.batch, class_mode='categorical')
        val_gen = val_aug.flow_from_directory(val_dir, target_size=IMG_SIZE, batch_size=args.batch, class_mode='categorical')
    else:
        train_aug = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.05,
            zoom_range=0.15,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest',
            validation_split=0.15
        )
        # Use subsets from the same folder
        train_gen = train_aug.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=args.batch, class_mode='categorical', subset='training')
        val_gen = train_aug.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=args.batch, class_mode='categorical', subset='validation')

    num_classes = len(train_gen.class_indices)

    # GPU / distributed strategy and mixed precision
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
        strategy = tf.distribute.MirroredStrategy()
        print(f'Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas')
        # Enable mixed precision for speed on GPUs
        try:
            mixed_precision.set_global_policy('mixed_float16')
            print('Enabled mixed precision policy: mixed_float16')
        except Exception:
            pass
    else:
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        model = build_model(input_shape=(IMG_SIZE[0],IMG_SIZE[1],3), num_classes=num_classes, backbone=args.backbone)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    os.makedirs(os.path.dirname(args.save_to), exist_ok=True)

    callbacks = [
        ModelCheckpoint(args.save_to, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=callbacks)

    # Evaluate on test set if available
    if os.path.exists(test_dir):
        test_gen = val_aug.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=args.batch, class_mode='categorical', shuffle=False)
        loss, acc = model.evaluate(test_gen)
        print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")

    # Save final model (already saved by checkpoint)
    model.save(args.save_to)
    print('Model saved to', args.save_to)

    # Save class name order so inference code can map indices back to labels
    class_list = [None] * num_classes
    for name, idx in train_gen.class_indices.items():
        class_list[idx] = name
    classes_path = os.path.join(os.path.dirname(args.save_to), 'class_indices.json')
    with open(classes_path, 'w', encoding='utf-8') as f:
        json.dump(class_list, f, ensure_ascii=False, indent=2)
    print('Saved class label order to', classes_path)
