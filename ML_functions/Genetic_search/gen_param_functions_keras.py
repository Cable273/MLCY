#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def create_population(count,variables,net_params):
    pop=np.zeros((count,variables))
    for network in range(0,count):
        for item in range(0,variables):
           pop[network,item] = net_params(item) 
    return pop

def mutate(network,variables,net_params):
    #mutate all variables with a % chance
    temp = np.zeros(np.size(network))
    mutate_chance = 0.1
    for count in range(0,np.size(network)):
        chance = np.random.uniform(0,1)
        if chance <= mutate_chance:
            temp[count] = net_params(count)
        else:
            temp[count]=network[count]
    return temp

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

#check populations to see if network already in it
#returns array, [0 or 1,index], 1 is true (exists in population)
# index is which entry of prev population it is
def check_existing_pop(network,check_pop):
    in_pop=0
    pop_index=0
    for count in range(0,np.size(check_pop,axis=0)):
        if np.array_equal(check_pop[count,:],network):
            in_pop = 1
            pop_index = count
            break
    return np.array((in_pop,pop_index))


def evolve(pop,scores,variables,net_params):
    # Evolve a list of network parameters into a new set
    # Train each network and keep the best perfroming ones (natural selection)
    fraction_winners=0.25
    loser_survival_chance=0.2

    pop_size=np.size(pop,axis=0)

    #sort by scores
    indexes=np.arange(0,pop_size)
    sorted_scores,indexes_sorted=zip( *sorted(zip(scores,indexes)))
    indexes_sorted=np.flip(indexes_sorted,axis=0)

    sorted_pop=np.zeros((pop_size,variables))
    for count in range(0,pop_size):
        sorted_pop[count]=pop[indexes_sorted[count]]

    #keep top fraction
    length_to_keep=int(np.ceil(fraction_winners*pop_size))
    next_gen=np.zeros((length_to_keep,variables))
    for count in range(0,length_to_keep):
        next_gen[count,:]=sorted_pop[count]

    #keep some other nets randomly
    for count in range(length_to_keep,pop_size):
        chance=np.random.uniform(0,1)
        if chance <= loser_survival_chance:
            next_gen=np.vstack((next_gen,sorted_pop[count]))

    #breed some children to fill rest
    remaining_to_fill = pop_size - np.size(next_gen,axis=0)
    children=np.zeros(variables)
    while np.size(children,axis=0)-1 < remaining_to_fill:
        #get parent networks
        mother_index  = np.random.choice((np.arange(0,np.size(next_gen,axis=0))))
        father_index  = np.random.choice((np.arange(0,np.size(next_gen,axis=0))))

        if np.array_equal(next_gen[mother_index,:],next_gen[father_index,:])==False:
            mother=next_gen[mother_index,:]
            father=next_gen[father_index,:]

            new_nets = breed(mother,father,variables)
            #Mutate children
            new_nets[0,:] = mutate(new_nets[0,:],variables,net_params)
            new_nets[1,:] = mutate(new_nets[1,:],variables,net_params)

            #avoid duplicate nets in population
            check_0 = check_existing_pop(new_nets[0,:],next_gen)
            check_1 = check_existing_pop(new_nets[1,:],next_gen)
            if check_0[0] == 0:
                children = np.vstack((children,new_nets[0,:]))
            if check_1[0] == 0:
                #dont overfill
                if np.size(children,axis=0)-1 < remaining_to_fill:
                    children = np.vstack((children,new_nets[1,:]))
            
    #delete initialization row
    children=np.delete(children,0,axis=0)
    next_gen=np.vstack((next_gen,children))
    return(next_gen)
