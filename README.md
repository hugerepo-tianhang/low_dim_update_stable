Open source 2 research projects for more future collaboration and work done on them

# Project 1, Low dimensional DRL search path

## Abstract
We found that the search path of some most used deep reinforcement learning algorithms (PPO, SAC) mostly falls in a low dimensional space (<10 dimension explains >98% of variance vs ambient parameter space of over 1000) for all tested tasks. Moreover I found that first PCA direction points to the optimal parameter with small error. (stable_baselines/low_dim_analysis directory)

## Application
The plane of first 2 PCA axis always captures about 85% of the variance of the search path, also this plane is one of the most informative 2d euclidean surface about the return structure in parameter space in hard locomotion tasks since a slightly off surface will have almost 0 variance of very low return. For example, a slightly off parameter will cause the agent lose balance and not able to complete the task at all. (jump, run etc). You can use the visualization to gain intuition of the difficulty of the task, is there a plateau, how big is the plateau, move the surface along the normal direction to have a 3d visualization of the parameter blob. Below is an example of return landscape of Hopper task in the parameter space of a PPO agent.
![Alt text](readme_pics_low_dim/parameterlandscape.jpg?raw=true "return landscape Hopper")

Example of first PCA direction points roughly to optimal parameter, notice that as the training goes on, the orange curve goes below 20 degree error.

![Alt text](readme_pics_low_dim/firstPCA_VS_optimal_error.jpg?raw=true "first PCA direction error")


Also includes various trials to utilize this knowledge to accelerate those algorithms including trials to quickly estimate the subspace with small error and run ES algorithm on that subspace, identify the first PCA direction and only search the cone given by the first PCA direction. However, in hard locomotion tasks, the estimated subspace is not accurate enough to contain near optimal parameter with high probability.

## Future work
It's interesting to have an understanding of why above phenomenon happens, even though it's hard to prove theorems about it. 

# Project 2, Physics model in model free DRL algorithms

## Abstract 
Identifying linear and non linear correlations between neurons in a trained deep neural networks in DRL algorithms and variables from lagrangian equations. (new_neuron_analysis directory). This is the trial to see whether model free DRL algorithms encodes implicit model of the environment in their single neurons ( physics model in locomotion tasks ).

## Examples
You can see a clear linear correlation from variables in mass matrix from lagrangian equations and neurons in the trained model.(For a review of lagrangian equations, see https://fab.cba.mit.edu/classes/865.18/design/optimization/dynamics_1.pdf)

![Alt text](readme_pics_lagrangian/M_index_155_VS_layer_1.0_neuron_142.0_linear_correlation-0.988610646842882_normalized_SSE_3.0919008977837885_Syy_136.51385498046875.jpg?raw=true "M155 VS neuron142")
![Alt text](readme_pics_lagrangian/M_index_1672_VS_layer_0.0_neuron_820.0_linear_correlation-0.9935707917551969_normalized_SSE_2.4381617239831384_Syy_190.22752380371094.jpg?raw=true "M1672 VS neuron820")


# Credits
Projects done with advisor Prof Karen Liu
