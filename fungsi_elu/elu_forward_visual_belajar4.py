import numpy as np
import matplotlib.pyplot as plt

def elu (x: np.array,alpha:float)-> np.array:
    return np.where(x>0,x,alpha*(np.exp(x)-1))

x = np.linspace(-4, 2, 100) 
y = elu(x,alpha=1)

plt.plot(x,y, label='elu')
plt.axhline(0,color='black',linestyle="--", linewidth=0.5)
plt.axvline(0,color='black',linestyle="--", linewidth=0.5)
plt.title("fungsi elu")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()
plt.show()