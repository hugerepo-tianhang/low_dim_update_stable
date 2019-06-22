from new_neuron_analysis.dir_tree_util import *
from subprocess import Popen, PIPE


proj_dir = get_project_dir()

cmd = ["scp", "-r", "hang@berlin.cc.gatech.edu:~/projects/low_dim_update_stable/new_neuron_analysis/result", proj_dir]
p = Popen(cmd, stdout=PIPE, stderr=PIPE)
stdout, stderr = p.communicate()