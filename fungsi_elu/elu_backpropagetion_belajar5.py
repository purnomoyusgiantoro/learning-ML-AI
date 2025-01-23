import numpy as np
import matplotlib.pyplot as plt

class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def backward(self, x: np.ndarray) -> np.ndarray:
        grad = np.where(x > 0, 1, self.alpha * np.exp(x))
        return grad
    
elu = ELU(alpha=1.0)
x = np.linspace(-4, 4, 100)
y = elu.forward(x)
dy_dx = elu.backward(x)

plt.plot(x, y, label='ELU')
plt.plot(x, dy_dx, label="Derivatif ELU", linestyle='--')
plt.title("Fungsi ELU dan Turunannya")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
