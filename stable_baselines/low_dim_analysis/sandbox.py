from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# load some test data for demonstration and plot a wireframe
X, Y, Z = axes3d.get_test_data(0.1)
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)

def rotate(angle):
    ax.view_init(azim=angle)

rot_animation = FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)

rot_animation.save('rotation.gif', dpi=80, writer='imagemagick')
