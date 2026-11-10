import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


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
        X_array = np.array(X, dtype=float)
        self.mean = np.mean(X_array, axis=0)
        self.std = np.std(X_array, axis=0)
        self.std = np.where(self.std == 0, 1e-8, self.std)
        return self

    def transform(self, X):
        X_array = np.array(X, dtype=float)
        return (X_array - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class KNN:

    def __init__(self, k=5, task="classification", distance="euclidean"):
        self.k = k
        self.task = task
        self.distance = distance
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)
        return self

    def _compute_distances(self, X):
        X = np.array(X, dtype=float)
        if self.distance == "euclidean":
            dists = np.sqrt(
                np.sum(
                    (self.X_train[None, :, :] - X[:, None, :]) ** 2, axis=-1
                )
            )
        elif self.distance == "manhattan":
            dists = np.sum(
                np.abs(self.X_train[None, :, :] - X[:, None, :]), axis=-1
            )
        return dists

    def predict(self, X):
        X = np.array(X, dtype=float)
        predictions = []
        batch_size = 250

        for i in range(0, len(X), batch_size):
            X_batch = X[i : i + batch_size]
            if self.distance == "euclidean":
                dists = np.sqrt(
                    np.sum(
                        (self.X_train[None, :, :] - X_batch[:, None, :]) ** 2,
                        axis=-1,
                    )
                )
            elif self.distance == "manhattan":
                dists = np.sum(
                    np.abs(self.X_train[None, :, :] - X_batch[:, None, :]),
                    axis=-1,
                )

            k_indices = np.argpartition(dists, self.k, axis=1)[:, : self.k]

            for row_idx, neighbors_idx in enumerate(k_indices):
                neighbor_labels = self.y_train[neighbors_idx]
                if self.task == "classification":
                    vals, counts = np.unique(
                        neighbor_labels, return_counts=True
                    )
                    pred = vals[np.argmax(counts)]
                elif self.task == "regression":
                    pred = np.mean(neighbor_labels)
                predictions.append(pred)

        return np.array(predictions)

data = pd.read_csv("Dry_Bean_Dataset.csv", sep=";", decimal=",")

X = data.drop(columns=["Class"])
y = data["Class"]
classes = np.unique(y)
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
y_encoded = y.map(class_to_idx)

X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(
    X, y_encoded, val_size=0.15, test_size=0.15, random_state=42
)

scaler = Scaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
experiments = {
    "Scaled + Euclidean": (X_train_scaled, X_val_scaled, "euclidean"),
    "Scaled + Manhattan": (X_train_scaled, X_val_scaled, "manhattan"),
    "Unscaled + Euclidean": (
        np.array(X_train),
        np.array(X_val),
        "euclidean",
    ),
    "Unscaled + Manhattan": (
        np.array(X_train),
        np.array(X_val),
        "manhattan",
    ),
}

val_accuracies = {exp: [] for exp in experiments}
val_f1s = {exp: [] for exp in experiments}

for exp_name, (xtr, xvl, dist) in experiments.items():
    for k in k_values:
        model = KNN(k=k, task="classification", distance=dist)
        model.fit(xtr, y_train)
        preds = model.predict(xvl)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="macro")
        val_accuracies[exp_name].append(acc)
        val_f1s[exp_name].append(f1)

plt.figure(figsize=(10, 6))
for exp_name in experiments:
    plt.plot(k_values, val_accuracies[exp_name], marker="o", label=exp_name)
plt.title("Accuracy by k")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

best_config = None
best_acc = -1
best_k = None

for exp_name in experiments:
    max_acc = max(val_accuracies[exp_name])
    if max_acc > best_acc:
        best_acc = max_acc
        best_config = exp_name
        best_k = k_values[np.argmax(val_accuracies[exp_name])]

print(f"Лучшая конфигурация по Validation: {best_config}, k = {best_k}")

final_configs = {
    "Scaled + Euclidean": (
        X_train_scaled,
        X_test_scaled,
        "euclidean",
        k_values[np.argmax(val_accuracies["Scaled + Euclidean"])],
    ),
    "Scaled + Manhattan": (
        X_train_scaled,
        X_test_scaled,
        "manhattan",
        k_values[np.argmax(val_accuracies["Scaled + Manhattan"])],
    ),
    "Unscaled + Euclidean": (
        np.array(X_train),
        np.array(X_test),
        "euclidean",
        k_values[np.argmax(val_accuracies["Unscaled + Euclidean"])],
    ),
    "Unscaled + Manhattan": (
        np.array(X_train),
        np.array(X_test),
        "manhattan",
        k_values[np.argmax(val_accuracies["Unscaled + Manhattan"])],
    ),
}

results = []
for name, (xtr, xts, dist, k_opt) in final_configs.items():
    model = KNN(k=k_opt, task="classification", distance=dist)
    model.fit(xtr, y_train)
    preds = model.predict(xts)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    results.append(
        {
            "Конфигурация": name,
            "k": k_opt,
            "Accuracy": round(acc, 4),
            "Macro F1": round(f1, 4),
        }
    )

df_results = pd.DataFrame(results)
print("\nТаблица метрик (Test Set):")
print(df_results.to_string(index=False))

best_dist = "euclidean" if "Euclidean" in best_config else "manhattan"
use_scaled = "Scaled" in best_config
x_tr_best = X_train_scaled if use_scaled else np.array(X_train)
x_ts_best = X_test_scaled if use_scaled else np.array(X_test)

best_model = KNN(k=best_k, task="classification", distance=best_dist)
best_model.fit(x_tr_best, y_train)
final_preds = best_model.predict(x_ts_best)

cm = confusion_matrix(y_test, final_preds)

plt.figure(figsize=(8, 7))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (Test Set) - {best_config} (k={best_k})")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

plt.ylabel("Истинный класс")
plt.xlabel("Предсказанный класс")
plt.tight_layout()
plt.show()