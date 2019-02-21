#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

from gen_param_functions import create_population
from gen_param_functions import mutate
from gen_param_functions import breed
from gen_param_functions import evolve

from gen_net_train import sgmd
from gen_net_train import train

pop=create_population(5,3)
np.savetxt('gen0',pop,fmt='%i ')
print(pop)
for gen in range(0,3):
    #evaluate each net
    scores=np.zeros(np.size(pop,axis=0))
    for count in range(0,np.size(pop,axis=0)):
        scores[count]=train(pop[count,0],pop[count,1],pop[count,2])

    pop=evolve(pop,scores,3)

    print(pop)
    np.savetxt('gen'+str(gen+1),pop,fmt='%i ')

