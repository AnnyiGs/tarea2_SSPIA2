import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

print("=" * 65)
print("  1. CARGA Y PREPROCESAMIENTO DE DATOS")
print("=" * 65)

df = pd.read_csv("DATASET-SALARY/salarios.csv")
print(f"\n✔ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"  Columnas: {list(df.columns)}")
print(f"\nPrimeras 3 filas:\n{df.head(3).to_string(index=False)}")
print(f"\nValores nulos:\n{df.isnull().sum().to_string()}")

df.dropna(inplace=True)
print(f"\n✔ Tras eliminar nulos: {df.shape[0]} filas")

TARGET_COL = "Salary"
CAT_COLS = ["Gender", "Education Level", "Job Title"]
NUM_COLS = ["Age", "Years of Experience"]
FEATURE_COLS = NUM_COLS + CAT_COLS

X_raw = df[FEATURE_COLS].copy()
y = df[TARGET_COL].values.astype(float)

label_encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    X_raw[col] = le.fit_transform(X_raw[col].astype(str))
    label_encoders[col] = le
    print(f"  LabelEncoded '{col}': {list(le.classes_)[:5]} ...")

X = X_raw.values.astype(float)