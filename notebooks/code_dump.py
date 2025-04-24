import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer

from utils import generate_muap

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from simulators import simulator_biomime6

num_dim = 6
prior = utils.BoxUniform(low=0.5 * torch.ones(num_dim), high=torch.ones(num_dim))

import torch
import torch.nn as nn
import torch.nn.functional as F

class SummaryNet_2D(nn.Module):
    def __init__(self):
        super().__init__()
        # 2D convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        # Maxpool layer that reduces 32x32 image to 4x4
        self.pool = nn.MaxPool2d(kernel_size=8, stride=8)
        # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
        self.fc = nn.Linear(in_features=16 * 40 * 12, out_features=16)

    def forward(self, x):
        # Let's reshape x as it comes in
        x = x.view(-1, 1, 320, 96)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 40 * 12)
        x = F.relu(self.fc(x))
        return x

embedding_net = SummaryNet_2D()

# Output of simulator_biomime6 is 320x96
# True structure is 32x10x96
# First, let's just ignore that the "intuitive" dimension of the conv operator is 3

from sbi.inference import prepare_for_sbi, simulate_for_sbi

# make a SBI-wrapper on the simulator object for compatibility
simulator_wrapper, prior = prepare_for_sbi(simulator_biomime6, prior)
# instantiate the neural density estimator
neural_posterior = utils.posterior_nn(
    model="maf", embedding_net=embedding_net, hidden_features=10, num_transforms=2
)

from sbi.inference import SNPE

# setup the inference procedure with the SNPE-C procedure
inference = SNPE(prior=prior, density_estimator=neural_posterior)
# run the inference procedure on one round and 10000 simulated data points
theta, x = simulate_for_sbi(simulator_wrapper, prior, num_simulations=10000)

density_estimator = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior(density_estimator)

x.shape

theta.shape



# Other methods are "SNLE" or "SNRE".
posterior = infer(simulator_biomime6, prior, method="SNPE", num_simulations=1000)

## ------------------------------------------------------------------------------------------------

import torch
import numpy as np
from pathlib import Path
HOME = Path("/rds/general/user/pm1222/home/hierarchical-npe")

import BioMime.utils.basics as bm_basics
import BioMime.models.generator as bm_gen

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = bm_basics.update_config(HOME.joinpath('biomime_weights', 'config.yaml'))

"""
Define the simulators for the local (6-parameter) and global+local (7-parameter)
BioMime models. 

"""

def initialize_generator(model_path, num_conds=None):
    if num_conds is not None:
        config['Model']['Generator']['num_conds'] = num_conds
    generator = bm_gen.Generator(config.Model.Generator)
    generator = bm_basics.load_generator(model_path, generator, 'cuda')
    generator = generator.to(device)
    return generator

def sample_biomime(generator, pars, truncate=False):
    if pars.ndim == 1:
        pars = pars[None, :]

    n_MU = pars.shape[0]
    sim_muaps = []

    for _ in range(10):
        cond = pars.to(device)
        sim = generator.sample(n_MU, cond.float(), cond.device)

        sim = sim.to("cpu")
        if n_MU == 1:
            sim = sim.permute(1, 2, 0).detach()
        else:
            sim = sim.permute(0, 2, 3, 1).detach()
        sim_muaps.append(sim)

    muap = np.array(sim_muaps).mean(0)
    if truncate:
        muap = muap[1:-1, 21:29, :]
    
    return torch.from_numpy(muap.flatten())

# Initialize the generators
BIOMIME6 = initialize_generator(HOME.joinpath('biomime_weights', 'model_linear.pth'))
BIOMIME7 = initialize_generator(HOME.joinpath('biomime_weights','biomime7_weights.pth'), num_conds=7)

# Define simulators
def simulator_biomime6(pars):
    return sample_biomime(BIOMIME6, pars)

def simulator_biomime7(pars):
    return sample_biomime(BIOMIME7, pars)

def simulator_biomime6_tr(pars):
    return sample_biomime(BIOMIME6, pars, True)

## ------------------------------------------------------------------------------------------------

import torch
from BioMime.utils.basics import update_config, load_model, load_generator
from BioMime.models.generator import Generator

import matplotlib.pyplot as plt
import seaborn as sns

from utils import plot_muap_simple

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

import numpy as np
import csv

# Load weights of Biomime 6 and 7
bm6 = torch.load('biomime_weights/model_linear.pth', torch.device('cpu'))
bm7 = torch.load('biomime_weights/epoch-8_checkpoint.pth', torch.device('cpu'))
bm6_keys = list(bm6.keys())
bm7_keys = list(bm7['generator'].keys())

# Sort keys so that mapping is easier
sortorder6 = np.argsort(bm6_keys)
sortorder7 = np.argsort(bm7_keys)

# Ensuring that biomime keys are properly ordered*
bm6_sizes = [str(bm6[k].shape) for k in bm6_keys]
bm7_sizes = [str(bm7['generator'][k].shape) for k in bm7_keys]

keys_unmapped = np.array([
    np.array(bm6_keys)[sortorder6],
    np.array(bm6_sizes)[sortorder6],
    np.array(bm7_keys)[sortorder7],
    np.array(bm7_sizes)[sortorder7],
]).T

with open('keys.csv', mode='w', newline='') as file:
    csv_writer = csv.writer(file, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(keys_unmapped)

# From here, I reordered thekeys manually in the csv and saved it at
# ../BioMime/ckp/keys_mapping_biomime6-7.csv
with open('biomime_weights/keys_mapping_biomime6-7.csv', mode='r', newline='') as file:
    csv_reader = csv.reader(file, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    keys_mapping = np.array(list(csv_reader))

# Now rename the bm7 keys according to bm6 convention
bm7_keys_old = np.array(bm7_keys)[sortorder7]
bm7_keys_new = list(keys_mapping[:, -2])
old_state_dict = bm7['generator']
new_state_dict = {}
for i in range(len(bm7_keys_old)):
    new_state_dict[bm7_keys_new[i]] = old_state_dict[bm7_keys_old[i]]

# Simple generation of MUAPs
config = update_config('biomime_weights/config.yaml')
config['Model']['Generator']['num_conds'] = 7
biomime7 = Generator(config.Model.Generator)
biomime7.load_state_dict(new_state_dict)

torch.save(biomime7.state_dict(), 'biomime_weights/biomime7_weights.pth')

config = update_config('biomime_weights/config.yaml')
biomime6 = Generator(config.Model.Generator)
biomime6.load_state_dict(bm6)

fdensity = widgets.FloatSlider(value = 0.75, min = 0.5, max = 1.0, step = 0.01)
depth = widgets.FloatSlider(value = 0.75, min = 0.5, max = 1.0, step = 0.01)
angle = widgets.FloatSlider(value = 0.75, min = 0.5, max = 1.0, step = 0.01)
izone = widgets.FloatSlider(value = 0.75, min = 0.5, max = 1.0, step = 0.01)
cvel = widgets.FloatSlider(value = 0.75, min = 0.5, max = 1.0, step = 0.01)
flength = widgets.FloatSlider(value = 0.75, min = 0.5, max = 1.0, step = 0.01)

def generate_plot_muap(fd, d, a, iz, cv, fl):
    n_MU = 1
    n_steps = 10
    z = torch.rand(n_MU, 16)  # Latent noise
    c6 = torch.tensor((fd, d, a, iz, cv, fl))[None, :]
    sim_muaps = []

    for _ in range(n_steps):
        sim = BIOMIME6.sample(n_MU, c6.float(), c6.device, z)
        sim = sim.permute(1, 2, 0).detach().numpy()
        sim_muaps.append(sim)

    sim_muaps = np.array(sim_muaps)
    mean_muap = np.mean(sim_muaps, axis=0)

    print(f'Average std across steps: {np.mean(np.std(sim_muaps, axis=0))}')
    plot_muap_simple(mean_muap[:, ::2, :])
    return None

interact_manual(generate_plot_muap, fd=fdensity, d=depth, a=angle, iz=izone, cv=cvel, fl=flength)
