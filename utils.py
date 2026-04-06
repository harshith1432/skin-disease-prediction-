# Utility helpers for dataset and evaluation
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def get_class_indices(generator):
    return {v:k for k,v in generator.class_indices.items()}


def evaluate_model(model, generator):
    preds = model.predict(generator)
    y_true = generator.classes
    y_pred = np.argmax(preds, axis=1)
    print(classification_report(y_true, y_pred, target_names=list(generator.class_indices.keys())))
    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))
