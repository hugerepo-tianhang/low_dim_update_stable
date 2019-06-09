import numpy as np
env = "DartWalker2d-v1"
num_timesteps = 10
out_dir = f"/home/panda-linux/PycharmProjects/low_dim_update_dart/low_dim_update_stable/neuron_vis/plots_{env}_{num_timesteps}"

ind = 1
fname = f"{out_dir}/weights_layer_{ind}.txt"
weights = np.loadtxt(fname)
pass

#TODO /usr/include$ ll eigen3/
# /usr/include$ ll bullet/
# total 48
# drwxr-xr-x   6 root root  4096 10月  2  2018 ./
# drwxr-xr-x 109 root root 20480 5月   7 09:27 ../
# -rw-r--r--   1 root root  3400 8月  11  2015 btBulletCollisionCommon.h
# -rw-r--r--   1 root root  2212 8月  11  2015 btBulletDynamicsCommon.h
# drwxr-xr-x   7 root root  4096 10月  2  2018 BulletCollision/
