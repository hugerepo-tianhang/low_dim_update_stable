import numpy as np

a = np.array([[[2,1,1],[3,2,2],[4,3,3]], [[5,4,4],[6,5,5],[7,6,6]]])

print(np.max(a, axis=1))
