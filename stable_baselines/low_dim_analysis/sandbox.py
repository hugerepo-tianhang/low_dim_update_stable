import numpy as np
from matplotlib import pyplot as plt
def f(a):
    return a

l = [f(i) for i in range(100)]
plt.plot(np.arange(100), l)
plt.show()