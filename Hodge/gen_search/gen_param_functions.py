#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#parameters to optimize
# number of layers
# neurons in each layer
# mini batch size

def net_params(item):
    if item == 0:
        return np.random.choice((1,2))
    if item == 1:
        return int(np.floor(np.random.uniform(10,200)))
    if item == 2:
        return int(np.floor(np.random.uniform(32,2048)))

def create_population(count,variables):
    pop=np.zeros((count,variables))
    for network in range(0,count):
        for item in range(0,variables):
           pop[network,item] = net_params(item) 
    return pop

def mutate(network,variables):
    # Randomly change one variable in network
    mutation_index = np.random.choice(np.arange(0,variables))
    network[mutation_index] = net_params(mutation_index) 
    return network

def breed(mother,father,variables):
    # Create two new networks from "parent" networks
    parents=np.vstack((mother,father))
    children=np.zeros((2,variables))

    for index in range(0,2):
        child=np.zeros(variables)
        for param in range(0,variables):
            child[param]=np.random.choice(parents[:,param])
        children[index,:]=child
    return children

def evolve(pop,scores,variables):
    # Evolve a list of network parameters into a new set
    # Train each network and keep the best perfroming ones (natural selection)
    mutate_chance=0.1
    loser_survival_chance=0.2

    pop_size=np.size(pop,axis=0)

    # accuracy=np.random.uniform(0,1,pop_size)

    #sort by scores
    indexes=np.arange(0,pop_size)
    sorted_scores,indexes_sorted=zip( *sorted(zip(scores,indexes)))
    indexes_sorted=np.flip(indexes_sorted,axis=0)

    sorted_pop=np.zeros((pop_size,variables))
    for count in range(0,pop_size):
        sorted_pop[count]=pop[indexes_sorted[count]]

    #keep top 25% of nets
    length_to_keep=int(np.ceil(0.25*pop_size))
    next_gen=np.zeros((length_to_keep,3))
    for count in range(0,length_to_keep):
        next_gen[count,:]=sorted_pop[count]

    #keep some other nets randomly
    for count in range(0,int(pop_size-length_to_keep)):
        chance=np.random.uniform(0,1)
        if chance <= loser_survival_chance:
            next_gen=np.vstack((next_gen,sorted_pop[count]))

    #breed some children to fill rest
    remaining_to_fill = pop_size - np.size(next_gen,axis=0)
    children=np.zeros(3)
    while np.size(children,axis=0)-1 < remaining_to_fill:
        #get parent networks
        mother_index  = np.random.choice((np.arange(0,np.size(next_gen,axis=0))))
        father_index  = np.random.choice((np.arange(0,np.size(next_gen,axis=0))))

        if mother_index != father_index:
            mother=next_gen[mother_index,:]
            father=next_gen[father_index,:]

            new_nets = breed(mother,father,variables)
            children = np.vstack((children,new_nets[0,:]))

            #dont overfill
            if np.size(children,axis=0)-1 < remaining_to_fill:
                children = np.vstack((children,new_nets[1,:]))
            
    #delete initialization row
    children=np.delete(children,0,axis=0)

    #mutate some of children
    for count in range(0,np.size(children,axis=0)):
        chance=np.random.uniform(0,1)
        if chance <= mutate_chance:
            children[count,:]=mutate(children[count,:],variables)

    next_gen=np.vstack((next_gen,children))
    return(next_gen)
