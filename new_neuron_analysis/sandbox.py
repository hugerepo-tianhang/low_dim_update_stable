import numpy as np
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

harvest = np.array([[0,1,1],[1,0,0],[0,0,0]])


fig, ax = plt.subplots()
im = ax.imshow(harvest)



ax.set_title("included M matrix")
fig.tight_layout()
plt.show()
