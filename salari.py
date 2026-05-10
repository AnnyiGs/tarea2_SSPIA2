import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
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

TARGET_COL   = "Salary"
CAT_COLS     = ["Gender", "Education Level", "Job Title"]
NUM_COLS     = ["Age", "Years of Experience"]
FEATURE_COLS = NUM_COLS + CAT_COLS

X_raw = df[FEATURE_COLS].copy()
y     = df[TARGET_COL].values.astype(float)

label_encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    X_raw[col] = le.fit_transform(X_raw[col].astype(str))
    label_encoders[col] = le
    print(f"  LabelEncoded '{col}': {list(le.classes_)[:5]} ...")

X        = X_raw.values.astype(float)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
print(f"\n✔ División 80/20:")
print(f"  Entrenamiento: {X_train.shape[0]} muestras")
print(f"  Prueba:        {X_test.shape[0]} muestras")
n_features = X_train.shape[1]


def build_mlp(n_features, hidden_layers=2, neurons=64, activation="relu", optimizer="adam", learning_rate=0.001):
    opt_map = {
        "adam":     keras.optimizers.Adam(learning_rate=learning_rate),
        "sgd":      keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        "rmsprop":  keras.optimizers.RMSprop(learning_rate=learning_rate),
        "adadelta": keras.optimizers.Adadelta(learning_rate=learning_rate),
    }
    opt   = opt_map.get(optimizer.lower(), keras.optimizers.Adam(learning_rate=learning_rate))
    model = keras.Sequential()
    model.add(layers.Input(shape=(n_features,)))
    for _ in range(hidden_layers):
        model.add(layers.Dense(neurons, activation=activation))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1))
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model


def train_and_evaluate(model, X_tr, y_tr, X_te, y_te, epochs=200, batch_size=32, label=""):
    es      = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=0)
    history = model.fit(X_tr, y_tr, validation_split=0.15, epochs=epochs,
                        batch_size=batch_size, callbacks=[es], verbose=0)
    y_pred_sc = model.predict(X_te, verbose=0).ravel()
    y_pred    = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
    y_real    = scaler_y.inverse_transform(y_te.reshape(-1, 1)).ravel()
    rmse      = np.sqrt(mean_squared_error(y_real, y_pred))
    mae       = mean_absolute_error(y_real, y_pred)
    r2        = 1 - np.sum((y_real - y_pred) ** 2) / np.sum((y_real - np.mean(y_real)) ** 2)
    print(f"  [{label:<22}] Epocas: {len(history.history['loss']):3d} | RMSE: ${rmse:>9,.2f} | MAE: ${mae:>9,.2f} | R2: {r2:.4f}")
    return history, rmse, mae, r2, y_pred, y_real


print("\n" + "=" * 65)
print("  3. PRUEBA DE OPTIMIZADORES  (2 capas, 64 neuronas, lr=0.001)")
print("=" * 65)

OPTIMIZERS  = ["adam", "sgd", "rmsprop", "adadelta"]
OPT_COLORS  = {"adam": "#2563EB", "sgd": "#DC2626", "rmsprop": "#16A34A", "adadelta": "#D97706"}
opt_results = {}

for opt_name in OPTIMIZERS:
    model = build_mlp(n_features, hidden_layers=2, neurons=64, optimizer=opt_name, learning_rate=0.001)
    hist, rmse, mae, r2, y_pred, y_real = train_and_evaluate(
        model, X_train, y_train, X_test, y_test, epochs=200, batch_size=32, label=opt_name.upper())
    opt_results[opt_name] = dict(history=hist, rmse=rmse, mae=mae, r2=r2, y_pred=y_pred, y_real=y_real)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Evolucion del Error por Optimizador\n(2 capas ocultas · 64 neuronas · lr = 0.001)",
             fontsize=14, fontweight="bold")
for ax, opt in zip(axes.flatten(), OPTIMIZERS):
    h, r2, rmse, col = opt_results[opt]["history"], opt_results[opt]["r2"], opt_results[opt]["rmse"], OPT_COLORS[opt]
    ax.plot(h.history["loss"],     label="Entrenamiento", color=col, linewidth=2)
    ax.plot(h.history["val_loss"], label="Validacion",    color=col, linewidth=2, linestyle="--", alpha=0.6)
    ax.set_title(f"Optimizador: {opt.upper()}\nR2 = {r2:.4f}   RMSE = ${rmse:,.0f}", fontsize=11)
    ax.set_xlabel("Epoca"); ax.set_ylabel("MSE (normalizado)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("DATASET-SALARY/fig_optimizadores.png", dpi=150, bbox_inches="tight")
plt.show()

opt_summary = pd.DataFrame([
    {"Optimizador": k.upper(), "RMSE ($)": f"{v['rmse']:,.2f}", "MAE ($)": f"{v['mae']:,.2f}", "R2": f"{v['r2']:.4f}"}
    for k, v in opt_results.items()
])
print("\n--- Resumen Optimizadores ---")
print(opt_summary.to_string(index=False))
best_opt = max(opt_results, key=lambda k: opt_results[k]["r2"])
print(f"\n✔ Mejor optimizador: {best_opt.upper()} (R2 = {opt_results[best_opt]['r2']:.4f})")


print("\n" + "=" * 65)
print(f"  4. AJUSTE DE HIPERPARAMETROS  (optimizador: {best_opt.upper()})")
print("=" * 65)

CONFIGS = [
    (1,  32,  0.001,  200, 32),
    (1,  64,  0.001,  200, 32),
    (2,  64,  0.001,  200, 32),
    (2, 128,  0.001,  200, 32),
    (3,  64,  0.001,  200, 32),
    (3, 128,  0.001,  200, 32),
    (2,  64,  0.01,   200, 32),
    (2,  64,  0.0001, 200, 32),
    (2,  64,  0.001,  200, 16),
    (2,  64,  0.001,  200, 64),
]
hp_records, hp_histories = [], []

for cfg in CONFIGS:
    hl, neu, lr, ep, bs = cfg
    lbl   = f"L={hl} N={neu} lr={lr} bs={bs}"
    model = build_mlp(n_features, hidden_layers=hl, neurons=neu, optimizer=best_opt, learning_rate=lr)
    hist, rmse, mae, r2, _, _ = train_and_evaluate(
        model, X_train, y_train, X_test, y_test, epochs=ep, batch_size=bs, label=lbl)
    hp_records.append({"Capas Ocultas": hl, "Neuronas/Capa": neu, "Learning Rate": lr,
                        "Batch Size": bs, "Epocas Reales": len(hist.history["loss"]),
                        "RMSE ($)": round(rmse, 2), "MAE ($)": round(mae, 2), "R2 (Accuracy)": round(r2, 4)})
    hp_histories.append((lbl, hist))

hp_df = pd.DataFrame(hp_records)
print("\n--- Tabla de Hiperparametros vs Metricas ---")
print(hp_df.to_string(index=False))
hp_df.to_csv("DATASET-SALARY/tabla_hiperparametros.csv", index=False)

best_idx   = hp_df["R2 (Accuracy)"].idxmax()
colors_bar = ["#2563EB" if i == best_idx else "#93C5FD" for i in range(len(hp_records))]
x_labels   = [f"L={r['Capas Ocultas']} N={r['Neuronas/Capa']}\nlr={r['Learning Rate']} bs={r['Batch Size']}"
               for r in hp_records]

fig2, ax = plt.subplots(figsize=(15, 5))
bars = ax.bar(range(len(hp_records)), hp_df["R2 (Accuracy)"], color=colors_bar, edgecolor="white", linewidth=0.8)
ax.set_xticks(range(len(hp_records))); ax.set_xticklabels(x_labels, fontsize=8)
ax.set_ylabel("R2 (Overall Accuracy)")
ax.set_title(f"Comparacion de Hiperparametros — R2 en Prueba\n(Optimizador: {best_opt.upper()})",
             fontsize=12, fontweight="bold")
ax.set_ylim(max(0, hp_df["R2 (Accuracy)"].min() - 0.05), 1.0)
ax.axhline(hp_df["R2 (Accuracy)"].max(), color="#DC2626", linestyle="--", linewidth=1, alpha=0.7, label="Mejor R2")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
for bar, val in zip(bars, hp_df["R2 (Accuracy)"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
plt.tight_layout()
plt.savefig("DATASET-SALARY/fig_hiperparametros.png", dpi=150, bbox_inches="tight")
plt.show()

fig3, axes3 = plt.subplots(2, 5, figsize=(20, 7))
fig3.suptitle(f"Curvas de Perdida — Ajuste de Hiperparametros ({best_opt.upper()})", fontsize=13, fontweight="bold")
for ax, (lbl, hist), r2 in zip(axes3.flatten(), hp_histories, hp_df["R2 (Accuracy)"]):
    ax.plot(hist.history["loss"],     color="#2563EB", linewidth=1.5, label="Train")
    ax.plot(hist.history["val_loss"], color="#DC2626", linewidth=1.5, linestyle="--", label="Val")
    ax.set_title(f"{lbl}\nR2={r2:.4f}", fontsize=8)
    ax.set_xlabel("Epoca", fontsize=7); ax.set_ylabel("MSE", fontsize=7)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("DATASET-SALARY/fig_curvas_hp.png", dpi=150, bbox_inches="tight")
plt.show()


print("\n" + "=" * 65)
print("  5. MODELO FINAL Y EVALUACION")
print("=" * 65)

hl_b, neu_b, lr_b, ep_b, bs_b = CONFIGS[best_idx]
print(f"\n✔ Mejor configuracion: L={hl_b} N={neu_b} lr={lr_b} bs={bs_b} opt={best_opt.upper()}")

final_model = build_mlp(n_features, hidden_layers=hl_b, neurons=neu_b, optimizer=best_opt, learning_rate=lr_b)
es_final    = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=0)
print("\nEntrenando modelo final...")
hist_final  = final_model.fit(X_train, y_train, validation_split=0.15, epochs=300,
                               batch_size=bs_b, callbacks=[es_final], verbose=1)

y_pred_sc  = final_model.predict(X_test, verbose=0).ravel()
y_pred_fin = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
y_real_fin = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
rmse_fin   = np.sqrt(mean_squared_error(y_real_fin, y_pred_fin))
mae_fin    = mean_absolute_error(y_real_fin, y_pred_fin)
r2_fin     = 1 - np.sum((y_real_fin - y_pred_fin) ** 2) / np.sum((y_real_fin - np.mean(y_real_fin)) ** 2)
print(f"\n  RMSE Final : ${rmse_fin:,.2f}")
print(f"  MAE Final  : ${mae_fin:,.2f}")
print(f"  R2 Final   : {r2_fin:.4f}")

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle(f"Evaluacion del Modelo Final  ({best_opt.upper()} · L={hl_b} · N={neu_b} · lr={lr_b} · bs={bs_b})",
              fontsize=13, fontweight="bold")
ax4a.plot(hist_final.history["loss"],     color="#2563EB", linewidth=2, label="Entrenamiento")
ax4a.plot(hist_final.history["val_loss"], color="#DC2626", linewidth=2, linestyle="--", label="Validacion")
ax4a.set_xlabel("Epoca"); ax4a.set_ylabel("MSE (normalizado)")
ax4a.set_title("Curva de Perdida — Entrenamiento vs Validacion")
ax4a.legend(); ax4a.grid(True, alpha=0.3)
ax4b.scatter(y_real_fin, y_pred_fin, alpha=0.45, color="#2563EB", s=18, label="Predicciones")
lims = [min(y_real_fin.min(), y_pred_fin.min()) * 0.95, max(y_real_fin.max(), y_pred_fin.max()) * 1.05]
ax4b.plot(lims, lims, "r--", linewidth=2, label="Ideal (y = x)")
ax4b.set_xlabel("Salario Real ($)"); ax4b.set_ylabel("Salario Predicho ($)")
ax4b.set_title(f"Salario Real vs Predicho\nR2 = {r2_fin:.4f}  |  RMSE = ${rmse_fin:,.0f}")
ax4b.legend(); ax4b.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("DATASET-SALARY/fig_evaluacion_final.png", dpi=150, bbox_inches="tight")
plt.show()


print("\n" + "=" * 65)
print("  6. PREDICCION PARA NUEVOS EMPLEADOS")
print("=" * 65)

df_pred = pd.read_excel("DATASET-SALARY/prediccion.xlsx")
df_pred = df_pred.dropna(axis=1, how="all").dropna(axis=0, how="all")
print(f"\n✔ prediccion.xlsx cargado: {df_pred.shape}")
print(df_pred.to_string(index=False))

DESIRED_COL      = "Desired salary"
desired_salaries = df_pred[DESIRED_COL].values.astype(float)
X_new            = df_pred[FEATURE_COLS].copy()

for col in CAT_COLS:
    le = label_encoders[col]
    def safe_encode(val, le=le, col=col):
        s = str(val)
        if s in le.classes_: return int(le.transform([s])[0])
        print(f"  ⚠ Clase '{s}' en '{col}' no vista. Usando fallback.")
        return 0
    X_new[col] = X_new[col].astype(str).apply(safe_encode)

X_new_scaled = scaler_X.transform(X_new.values.astype(float))
y_new_pred   = scaler_y.inverse_transform(
    final_model.predict(X_new_scaled, verbose=0).ravel().reshape(-1, 1)).ravel()

result_rows = []
for i, (pred, des) in enumerate(zip(y_new_pred, desired_salaries)):
    diff_pct = (pred - des) / des * 100
    result_rows.append({"Empleado": i + 1, "Salario Predicho ($)": round(pred, 2),
                         "Salario Deseado ($)": round(des, 2), "Diferencia ($)": round(pred - des, 2),
                         "Diferencia (%)": round(diff_pct, 2)})

results_df = pd.DataFrame(result_rows)
print("\n--- Predicciones vs Salario Deseado ---")
print(results_df.to_string(index=False))
results_df.to_csv("DATASET-SALARY/predicciones_nuevos.csv", index=False)

emp_labels = [f"Emp {i+1}" for i in range(len(y_new_pred))]
x_pos      = np.arange(len(emp_labels))
width      = 0.35
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5))
fig5.suptitle("Predicciones vs Salario Deseado — 5 Nuevos Empleados", fontsize=13, fontweight="bold")
bars1 = ax5a.bar(x_pos - width/2, y_new_pred,       width, label="Predicho", color="#2563EB")
bars2 = ax5a.bar(x_pos + width/2, desired_salaries, width, label="Deseado",  color="#DC2626")
ax5a.set_xticks(x_pos); ax5a.set_xticklabels(emp_labels)
ax5a.set_ylabel("Salario ($)"); ax5a.set_title("Salario Predicho vs Deseado")
ax5a.legend(); ax5a.grid(True, axis="y", alpha=0.3)
for bar in list(bars1) + list(bars2):
    ax5a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 400,
              f"${bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7)
diff_pcts   = results_df["Diferencia (%)"].values
colors_diff = ["#16A34A" if d >= 0 else "#DC2626" for d in diff_pcts]
ax5b.bar(emp_labels, diff_pcts, color=colors_diff, edgecolor="white")
ax5b.axhline(0, color="black", linewidth=1)
ax5b.set_ylabel("Diferencia (%)")
ax5b.set_title("% Diferencia: (Predicho − Deseado) / Deseado × 100")
ax5b.grid(True, axis="y", alpha=0.3)
for i, val in enumerate(diff_pcts):
    ax5b.text(i, val + (0.4 if val >= 0 else -1.5), f"{val:+.2f}%",
              ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig("DATASET-SALARY/fig_predicciones_nuevos.png", dpi=150, bbox_inches="tight")
plt.show()


print("\n" + "=" * 65)
print("  7. RESUMEN FINAL")
print("=" * 65)
print("\n--- Resumen Optimizadores ---")
print(opt_summary.to_string(index=False))
print("\n--- Mejor Configuracion ---")
print(hp_df.loc[[best_idx]].to_string(index=False))
print(f"\n--- Metricas Modelo Final ---")
print(f"  RMSE: ${rmse_fin:,.2f}  |  MAE: ${mae_fin:,.2f}  |  R2: {r2_fin:.4f}")
print("\n--- Resultados Nuevos Empleados ---")
print(results_df.to_string(index=False))