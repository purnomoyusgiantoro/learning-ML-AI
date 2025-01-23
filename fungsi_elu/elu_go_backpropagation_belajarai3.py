import numpy as np
import matplotlib.pyplot as plt

class ELU:
    def __init__(self, alpha=1.0):
        assert alpha > 0
        self.alpha = alpha
        
    def forward(self, x: np.ndarray)-> np.ndarray:
        return np.where(x > 0, x, self.alpha * (np.exp(x) -1))

    def plot(self,x_range=(-4,4),num_points=100):
            
        x = np.linspace(x_range[0],x_range[1],num_points)
        y = self.forward(x)

        plt.plot(x, y,label=f'elu(alpha={self.alpha})')
        plt.axhline(0,color='black',linestyle="--",linewidth=0.5)
        plt.title('fungsi elu')
        plt.xlabel('input x')
        plt.ylabel('output y')
        plt.legend()
        plt.grid()
        plt.show()

elu = ELU(alpha=1.0)
elu.plot()


