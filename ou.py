import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Activation functions
x = np.linspace(-10, 10, 400)
acts = {
    "Sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "Tanh": np.tanh,
    "ReLU": lambda x: np.maximum(0, x),
    "Leaky ReLU": lambda x: np.where(x > 0, x, 0.01 * x)
}

# Linear separability check
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)
clf = LogisticRegression().fit(X, y)


# Plot activation functions
plt.figure(figsize=(12, 8))
for i, (name, func) in enumerate(acts.items(), 1):
    plt.subplot(2, 3, i)
    plt.plot(x, func(x))
    plt.title(name)
    plt.grid(True)

# Softmax
x_s = np.linspace(-2, 2, 5)
smax = np.exp(x_s - np.max(x_s)); smax /= smax.sum()
plt.subplot(2, 3, 5)
plt.bar(range(len(smax)), smax)
plt.title("Softmax")
plt.tight_layout()
plt.show()
