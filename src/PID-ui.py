from Controller.ClassicalController import ClassicalController
from Plants import BathtubPlant


# Pivotal Prameters

# System params
epochs = 10
n_timesteps = 100
lr = 0.01

# Controller param.
controller = ClassicalController()

# Plant releated Params
A=1.5
C=0.015
H_0=0.5
D_range = [-0.01, 0.01]
plant = BathtubPlant()



