# kfold_pipeline.py
import os
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
import numpy as np
import cv2
from datetime import datetime


class CSVImageGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, image_size=(224, 224), batch_size=64, shuffle=True, augment=False, num_classes=None):
        self.df = dataframe.copy()
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.num_classes = num_classes or self.df['Label'].nunique()
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.df['Label'].unique()))}
        self.samples = len(self.df)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        batch_df = self.df.iloc[index * self.batch_size: (index + 1) * self.batch_size]
        X, y = [], []

        for _, row in batch_df.iterrows():
            image = cv2.imread(row['Image_path'])
            if image is None:
                print(f"[AVISO] Imagem não carregada: {row['Image_path']}")
                continue
            image = cv2.resize(image, self.image_size)
            image = image / 255.0
            X.append(image)
            y.append(self.label_map[row['Label']])

        return np.array(X), tf.keras.utils.to_categorical(y, num_classes=self.num_classes)


def expand_image_paths(df):
    expanded_rows = []
    for _, row in df.iterrows():
        folder_path = row['Image_path']
        label = row['Label']
        if not os.path.isdir(folder_path):
            print(f"[AVISO] Caminho não encontrado: {folder_path}")
            continue
        for fname in os.listdir(folder_path):
            full_path = os.path.join(folder_path, fname)
            if os.path.isfile(full_path):
                expanded_rows.append({'Image_path': full_path, 'Label': label})
    return pd.DataFrame(expanded_rows)


def generate_folds(csv_path, k=5, seed=42, output_dir="outputs/folds", split_ratios=(0.7, 0.2, 0.1)):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"split_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    df_expanded = expand_image_paths(df)

    print(f"[INFO] Salvando arquivos na pasta: {output_dir}")

    trainval_df, test_df = train_test_split(df_expanded, test_size=split_ratios[2], stratify=df_expanded['Label'], random_state=seed)
    test_path = os.path.join(output_dir, "test.csv")
    test_df.to_csv(test_path, index=False)
    print(f"[INFO] Teste salvo: {test_path}")

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(trainval_df)):
        train_path = os.path.join(output_dir, f"fold_{fold_idx}_train.csv")
        val_path = os.path.join(output_dir, f"fold_{fold_idx}_val.csv")
        trainval_df.iloc[train_idx].to_csv(train_path, index=False)
        trainval_df.iloc[val_idx].to_csv(val_path, index=False)
        print(f"[INFO] Fold {fold_idx} salvo: {train_path}, {val_path}")


def get_csv_generators(train_csv_path, val_csv_path, test_csv_path=None, image_size=(224, 224), batch_size=64, num_classes=None):
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    train_generator = CSVImageGenerator(train_df, image_size=image_size, batch_size=batch_size, shuffle=True, augment=True, num_classes=num_classes)
    val_generator = CSVImageGenerator(val_df, image_size=image_size, batch_size=batch_size, shuffle=False, num_classes=num_classes)

    if test_csv_path:
        test_df = pd.read_csv(test_csv_path)
        test_generator = CSVImageGenerator(test_df, image_size=image_size, batch_size=batch_size, shuffle=False, num_classes=num_classes)
    else:
        test_generator = None

    return train_generator, val_generator, test_generator



class EpochTimer(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.start_time
        print(f"[TIMER] Epoch {epoch + 1} completed in {duration:.2f} seconds.")
