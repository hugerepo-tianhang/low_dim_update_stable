import matplotlib.pyplot as plt
import numpy as np
import minepy
def compute_alpha(npoints):
    NPOINTS_BINS = [1, 25, 50, 250, 500, 1000, 2500, 5000, 10000, 40000]
    ALPHAS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4]

    if npoints < 1:
        raise ValueError("the number of points must be >=1")

    return ALPHAS[np.digitize([npoints], NPOINTS_BINS)[0] - 1]

def mic(x,y):
    alpha_cl = compute_alpha(x.shape[0])
    mine = minepy.MINE(alpha=alpha_cl, c=5, est="mic_e")
    mine.compute_score(x, y)
    mic = mine.mic()
    return mic

Fs = 8000
f = 5
sample = 8000
x = np.arange(sample)
# y = np.sin(2 * np.pi * f * x / Fs)
# z = np.sin(2 * np.pi * f * x / Fs + np.pi/2)
# # plt.figure()
# # plt.plot(x, y)
# # plt.savefig("first_sin.jpg")
# # plt.figure()
# # plt.plot(x, z)
# # plt.savefig("second_sin.jpg")
#
out_dir = "/home/panda-linux/PycharmProjects/low_dim_update_dart/low_dim_update_stable/neuron_vis"
#
# plt.figure()
# plt.plot(y, z)
# plt.savefig(f"{out_dir}/first_VS_second_pi_div_2.jpg")
#
# print(mic(y,z))

x = np.arange(sample)
y = np.sin(2 * np.pi * f * x / Fs)
z = np.sin(4 * np.pi * f * x / Fs)
plt.figure()
plt.plot(x, y)
plt.savefig("first_sin.jpg")
plt.figure()
plt.plot(x, z)
plt.savefig("second_sin.jpg")


plt.figure()
plt.plot(y, z)
plt.savefig(f"{out_dir}/first_VS_second_double_freq.jpg")

print(mic(y,z))
