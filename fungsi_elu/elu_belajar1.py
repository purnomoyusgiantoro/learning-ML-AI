import numpy as np
import matplotlib.pyplot as plt

def elu(x, alpha:float):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

x = np.linspace(-3, 3, 100)
y = elu(x, alpha=1.0)

plt.plot(x, y)
plt.title("ELU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()
