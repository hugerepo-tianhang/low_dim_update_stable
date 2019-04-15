from stable_baselines.low_dim_analysis.common import cal_angle_between_nd_planes
import numpy as np
pcs1 = np.array([[0,1,0],[1,0,1]])
pcs2 = np.array([[1,1,1],[-1,1,-1]])
print(cal_angle_between_nd_planes(pcs1, pcs2))