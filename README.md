Code for two reasearch projects.

1. Identifying a low dimensional space of the search path of most used deep reinforcement learning algorithms and found that first PCA points to optimal parameter with small error. Also includes various trials to utilize this knowledge to accelerate those algorithms including trials to identify the subspace with small error and run ES algorithm on that subspace, identify the first PCA direction and only search the cone given by the first PCA direction. (stable_baselines/low_dim_analysis directory)
2. Identifying linear and non linear correlations from neurons in a trained deep neural networks in DRL algorithms and lagrangian variables from classical control. Also includes trials to accelerate algorthms using this knowledge.(new_neuron_analysis directory)
