"""
Precompute embeddings for images under model/train and model/test using MobileNetV2 backbone.
Saves `model/embeddings.npy` and `model/embeddings_meta.json`.
Run once after placing dataset in `model/train` and `model/test`.
"""
import os
import json
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'model')
TRAIN_DIR = os.path.join(MODEL_DIR, 'train')
TEST_DIR = os.path.join(MODEL_DIR, 'test')
OUT_EMB = os.path.join(MODEL_DIR, 'embeddings.npy')
OUT_META = os.path.join(MODEL_DIR, 'embeddings_meta.json')

IMG_SIZE = (224,224)


def iter_images(folder):
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                yield os.path.join(root, f)


def main():
    # collect image paths
    paths = []
    if os.path.exists(TRAIN_DIR):
        paths.extend(list(iter_images(TRAIN_DIR)))
    if os.path.exists(TEST_DIR):
        paths.extend(list(iter_images(TEST_DIR)))

    if not paths:
        print('No images found in', TRAIN_DIR, 'or', TEST_DIR)
        return

    # load backbone
    backbone = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE[0],IMG_SIZE[1],3))

    embeddings = []
    meta = []
    batch = []
    batch_paths = []
    BATCH_SIZE = 32

    for p in paths:
        try:
            img = Image.open(p).convert('RGB').resize(IMG_SIZE)
            arr = np.array(img)
            arr = preprocess_input(arr)
            batch.append(arr)
            batch_paths.append(p)
        except Exception as e:
            print('skip', p, e)
            continue

        if len(batch) >= BATCH_SIZE:
            x = np.stack(batch, axis=0)
            emb = backbone.predict(x, verbose=0)
            embeddings.append(emb)
            for bp in batch_paths:
                label = os.path.relpath(bp, MODEL_DIR).split(os.sep)[1] if os.path.relpath(bp, MODEL_DIR).count(os.sep)>=1 else ''
                meta.append({'path': bp, 'label': label})
            batch = []
            batch_paths = []

    if batch:
        x = np.stack(batch, axis=0)
        emb = backbone.predict(x, verbose=0)
        embeddings.append(emb)
        for bp in batch_paths:
            label = os.path.relpath(bp, MODEL_DIR).split(os.sep)[1] if os.path.relpath(bp, MODEL_DIR).count(os.sep)>=1 else ''
            meta.append({'path': bp, 'label': label})

    embeddings = np.vstack(embeddings)
    # normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    embeddings = embeddings / norms

    np.save(OUT_EMB, embeddings)
    with open(OUT_META, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('Saved', OUT_EMB, 'and', OUT_META, 'for', len(meta), 'images')


if __name__ == '__main__':
    main()
