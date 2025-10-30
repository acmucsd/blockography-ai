import os
import cv2
import numpy as np
import pandas as pd

def load_train_data(data_dir='../train'):
    X = []
    y = []
    for biome_name in os.listdir(data_dir):
        biome_path = os.path.join(data_dir, biome_name)
        if not os.path.isdir(biome_path):
            continue
        for img_name in os.listdir(biome_path):
            img_path = os.path.join(biome_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            X.append(img)
            y.append(biome_name)
    return X, y

def load_eval_data(data_dir='../eval'):
    X = []
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        X.append(img)
    return X
    
def save_predictions(predictions, output_file='predictions.csv'):
    df = pd.DataFrame({"id": np.arange(1, len(predictions) + 1), "prediction": predictions})
    df.to_csv(output_file, index=False)
    print(f"Saved {output_file}")
    return