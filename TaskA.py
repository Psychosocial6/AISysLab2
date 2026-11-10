import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_test_split(X, y, val_size = 0.15, test_size=0.15, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    test_end = int(n * test_size)
    val_end = test_end + int(n * val_size)
    test_indices = indices[:test_end]
    val_indices = indices[test_end:val_end]
    train_indices = indices[val_end:]
    if hasattr(X, "iloc"):
        X_train, X_val, X_test = X.iloc[train_indices], X.iloc[val_indices], X.iloc[test_indices]
    else:
        X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
    if hasattr(y, "iloc"):
        y_train, y_val, y_test = y.iloc[train_indices], y.iloc[val_indices], y.iloc[test_indices]
    else:
        y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test


class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X_array = np.array(X)
        self.mean = np.mean(X_array, axis=0)
        self.std = np.std(X_array, axis=0)
        self.std = np.where(self.std == 0, 1e-8, self.std)
        return self

    def transform(self, X):
        X_array = np.array(X)
        return (X_array - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, penalty=None, lam=0.0, l1_r=0.5):
        self.lr = learning_rate
        self.iterations = iterations
        self.penalty = penalty
        self.lam = lam
        self.l1_r = l1_r
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)
        samples, features = X.shape
        self.weights = np.zeros((features, 1))
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            error = y_predicted - y
            dw = (2 / samples) * np.dot(X.T, error)
            db = (2 / samples) * np.sum(error)
            mse_loss = np.mean(error ** 2)
            reg_loss = 0.0

            if self.penalty == "l1":
                reg_loss = self.lam * np.sum(np.abs(self.weights))
                dw += self.lam * np.sign(self.weights)
            elif self.penalty == "l2":
                reg_loss = self.lam * np.sum(self.weights ** 2)
                dw += 2 * self.lam * self.weights
            elif self.penalty == "elasticnet":
                l1 = self.l1_r * np.sum(np.abs(self.weights))
                l2 = (1 - self.l1_r) * np.sum(self.weights ** 2)
                reg_loss = self.lam * (l1 + l2)
                dw += self.lam * (self.l1_r * np.sign(self.weights) + 2 * (1 - self.l1_r) * self.weights)

            loss = mse_loss + reg_loss
            self.loss_history.append(loss)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        return (np.dot(X, self.weights) + self.bias).flatten()


data = pd.read_csv('day.csv')
drop_cols = ['instant', 'dteday', 'casual', 'registered', 'cnt']
X = data.drop(columns=drop_cols)
y = data['cnt']
names = X.columns.tolist()
X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y, val_size=0.15, test_size=0.15, random_state=42)
scaler = Scaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

lams = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0]
l1_rs = [0.2, 0.5, 0.7]
best = {"None" : {"lam" : 0.0}}

best_l1_mse = float("inf")
best_l1_lam = None
for l in lams:
    m = LinearRegression(
        learning_rate=0.02, iterations=500, penalty="l1", lam=l
    )
    m.fit(X_train_scaled, y_train)
    val_mse = mean_squared_error(y_val, m.predict(X_val_scaled))
    if val_mse < best_l1_mse:
        best_l1_mse = val_mse
        best_l1_lam = l
best["L1"] = {"lam": best_l1_lam}

best_l2_mse = float("inf")
best_l2_lam = None
for l in lams:
    m = LinearRegression(
        learning_rate=0.02, iterations=500, penalty="l2", lam=l
    )
    m.fit(X_train_scaled, y_train)
    val_mse = mean_squared_error(y_val, m.predict(X_val_scaled))
    if val_mse < best_l2_mse:
        best_l2_mse = val_mse
        best_l2_lam = l
best["L2"] = {"lam": best_l2_lam}

best_el_mse = float("inf")
best_el_config = None
for l in lams:
    for r in l1_rs:
        m = LinearRegression(
            learning_rate=0.02,
            iterations=500,
            penalty="elasticnet",
            lam=l,
            l1_r=r,
        )
        m.fit(X_train_scaled, y_train)
        val_mse = mean_squared_error(y_val, m.predict(X_val_scaled))
        if val_mse < best_el_mse:
            best_el_mse = val_mse
            best_el_config = (l, r)
best["ElasticNet"] = {
    "lam": best_el_config[0],
    "l1_r": best_el_config[1],
}

print("Оптимальные параметры:")
for k, v in best.items():
    print(f"{k}: {v}")

LR = 0.02
ITERS = 500

models = {
    "None": LinearRegression(
        learning_rate=LR, iterations=ITERS, penalty=None
    ),
    "L1": LinearRegression(
        learning_rate=LR,
        iterations=ITERS,
        penalty="l1",
        lam=best["L1"]["lam"],
    ),
    "L2": LinearRegression(
        learning_rate=LR,
        iterations=ITERS,
        penalty="l2",
        lam=best["L2"]["lam"],
    ),
    "ElasticNet": LinearRegression(
        learning_rate=LR,
        iterations=ITERS,
        penalty="elasticnet",
        lam=best["ElasticNet"]["lam"],
        l1_r=best["ElasticNet"]["l1_r"],
    ),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

plt.figure(figsize=(10, 6))
for name, model in models.items():
    plt.plot(model.loss_history, label=f"{name}")
plt.title("Сравнение сходимости")
plt.xlabel("Итерация")
plt.ylabel("Loss")
plt.yscale("log")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
bar_width = 0.2
indices = np.arange(len(names))

for i, (name, model) in enumerate(models.items()):
    plt.bar(
        indices + i * bar_width,
        model.weights.flatten(),
        width=bar_width,
        label=name,
    )
plt.xticks(indices + bar_width * 1.5, names, rotation=45, ha="right")
plt.title("Сравнение весов")
plt.xlabel("Признак")
plt.ylabel("Вес")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

results = []
eps = 1e-2

for name, model in models.items():
    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    w = model.weights.flatten()
    zero_cnt = np.sum(np.abs(w) < eps)
    weights_l2_norm = np.linalg.norm(w)

    results.append(
        {
            "Модель": name,
            "Параметры": (
                best[name] if name != "None" else "lam=0"
            ),
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
            "||w||": round(weights_l2_norm, 2),
            f"Почти нулевые веса:": zero_cnt,
        }
    )

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))