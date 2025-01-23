import numpy as np

def elu (x: np.array,alpha:float)-> np.array:
    return np.where(x>0,x,alpha*(np.exp(x)-1))

array= np.array([-2,-2,-4,3])

print(array)
print(elu(array,alpha=1))

