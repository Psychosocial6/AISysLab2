import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def train_test_split(X, y, val_size=0.15, test_size=0.15, random_state=42):
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
        X_train, X_val, X_test = (
            X.iloc[train_indices],
            X.iloc[val_indices],
            X.iloc[test_indices],
        )
    else:
        X_train, X_val, X_test = (
            X[train_indices],
            X[val_indices],
            X[test_indices],
        )
    if hasattr(y, "iloc"):
        y_train, y_val, y_test = (
            y.iloc[train_indices],
            y.iloc[val_indices],
            y.iloc[test_indices],
        )
    else:
        y_train, y_val, y_test = (
            y[train_indices],
            y[val_indices],
            y[test_indices],
        )

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


class LogisticRegression:
    def __init__(
        self,
        learning_rate=0.1,
        iterations=500,
        penalty=None,
        lam=0.0,
        l1_r=0.5,
    ):
        self.lr = learning_rate
        self.iterations = iterations
        self.penalty = penalty
        self.lam = lam
        self.l1_r = l1_r
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def compute_loss(self, y_true, y_predicted_prob):
        eps = 1e-15
        y_predicted_prob = np.clip(y_predicted_prob, eps, 1.0 - eps)
        return -np.mean(
            y_true * np.log(y_predicted_prob)
            + (1.0 - y_true) * np.log(1.0 - y_predicted_prob)
        )

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)
        samples, features = X.shape
        self.weights = np.zeros((features, 1))
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted_prob = self.sigmoid(linear_model)

            dw = (1 / samples) * np.dot(X.T, (y_predicted_prob - y))
            db = (1 / samples) * np.sum(y_predicted_prob - y)

            base_loss = self.compute_loss(y, y_predicted_prob)
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

            loss = base_loss + reg_loss
            self.loss_history.append(loss)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_prob(self, X):
        X = np.array(X, dtype=float)
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model).flatten()

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_prob(X)
        return (probabilities >= threshold).astype(int)


data = pd.read_csv("train.csv")
drop_cols = ["PassengerId", "Name", "Cabin"]
df_clean = data.drop(columns=drop_cols).copy()
num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]

for col in num_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

for col in cat_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

y = df_clean["Transported"].astype(int)
X = df_clean.drop(columns=["Transported"])
X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)
names = X.columns.tolist()

X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(
    X, y, val_size=0.15, test_size=0.15, random_state=42
)

scaler = Scaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

lams = [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0]
l1_rs = [0.2, 0.5, 0.7]
best = {"None": {"lam": 0.0}}

best_l1_loss = float("inf")
best_l1_lam = None
for l in lams:
    m = LogisticRegression(
        learning_rate=0.1, iterations=1500, penalty="l1", lam=l
    )
    m.fit(X_train_scaled, y_train)
    val_loss = log_loss(y_val, m.predict_prob(X_val_scaled))
    if val_loss < best_l1_loss:
        best_l1_loss = val_loss
        best_l1_lam = l
best["L1"] = {"lam": best_l1_lam}

best_l2_loss = float("inf")
best_l2_lam = None
for l in lams:
    m = LogisticRegression(
        learning_rate=0.1, iterations=1500, penalty="l2", lam=l
    )
    m.fit(X_train_scaled, y_train)
    val_loss = log_loss(y_val, m.predict_prob(X_val_scaled))
    if val_loss < best_l2_loss:
        best_l2_loss = val_loss
        best_l2_lam = l
best["L2"] = {"lam": best_l2_lam}

best_el_loss = float("inf")
best_el_config = None
for l in lams:
    for r in l1_rs:
        m = LogisticRegression(
            learning_rate=0.1,
            iterations=1500,
            penalty="elasticnet",
            lam=l,
            l1_r=r,
        )
        m.fit(X_train_scaled, y_train)
        val_loss = log_loss(y_val, m.predict_prob(X_val_scaled))
        if val_loss < best_el_loss:
            best_el_loss = val_loss
            best_el_config = (l, r)
best["ElasticNet"] = {
    "lam": best_el_config[0],
    "l1_r": best_el_config[1],
}

print("Оптимальные параметры:")
for k, v in best.items():
    print(f"{k}: {v}")

LR = 0.1
ITERS = 1500

models = {
    "None": LogisticRegression(
        learning_rate=LR, iterations=ITERS, penalty=None
    ),
    "L1": LogisticRegression(
        learning_rate=LR,
        iterations=ITERS,
        penalty="l1",
        lam=best["L1"]["lam"],
    ),
    "L2": LogisticRegression(
        learning_rate=LR,
        iterations=ITERS,
        penalty="l2",
        lam=best["L2"]["lam"],
    ),
    "ElasticNet": LogisticRegression(
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
    probabilities = model.predict_prob(X_test_scaled)
    predictions = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    loss_val = log_loss(y_test, probabilities)

    w = model.weights.flatten()
    zero_cnt = np.sum(np.abs(w) < eps)
    weights_l2_norm = np.linalg.norm(w)

    results.append(
        {
            "Модель": name,
            "Параметры": (best[name] if name != "None" else "lam=0"),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "LogLoss": round(loss_val, 4),
            "||w||": round(weights_l2_norm, 2),
            "Почти нулевые веса:": zero_cnt,
        }
    )

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))