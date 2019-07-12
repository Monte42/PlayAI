# Libraies Neede

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# The Neural Network

class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
#Below we make full connection between layers in the network.  The first is a
# connection between the input layer and the hidden layer, which you can have as
# many as needed(try differ layers). The second connects the hidden layer to the
# possible actions. Linear(num of inputs, num of possible outputs, bias = True)
        self.fc1 = nn.Linear(input_size, 30)
#ABOVE you can change the second param to any num, this creates nerons that the
# linerar func will give values too for the ai to compare in order to make a choice.
#There is a third param "bias = True" leave as defualt
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):
# Here we use F from above and the relu func to activate the hidden neurons.
# The state param represents the vals of the input_size in the ai current posistion
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

#Using Experience Replay, we create a class that holds the memory(lis) of a set
# number of the last events that took place in that current state(pos)

class ReplayMemory(object):
    def __init__(self, capacity):
#The init func will allow you to adjust the lenght of the memory and then
# initialize the memory or create an empty list
        self.capacity = capacity
        self.memory = []

# The push func will add new events to the list and keep the list at below capacity
    def push(self, event):
#The event consists of four elements, last state(st), new state(st + 1), last action
# taken(at), and the last reward(rt)
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            # below will delete the first event when capacity is exceeded
            del self.memory[0]

# the sample func will collect random past samples of events in that state
    def sample(self, batch_size):
# the batch_size is adjustable, it is the number of samples collected
        samples = zip(*random.sample(self.memory, batch_size))
# what happend here is the zip(*) func takes a group of list with equal elements
# and combines them in a new list(samples), but reformats the list so each index
# from list is combind in a tuple then pushed to the list. this is done so we have
# a list of four tuples containing specific 'event' elemtents ex. last stae(st)
#the random.sample() func take random indexes from the memory.
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Deep Q learing

class Dqn():

    def __init__(self, input_size, nb_action, gamma):
# Gamma is the delay coefficent used in the formula for deep q learning.  Usally
# a num between 0 and 1, its multi with the next outputs and added to the batch
# of random rewards from the memory on that state.  this is used to make the next choice
        self.gamma = gamma
# reward_window will be a evolving list, "sliding window", of the meaning of the
# last 100 events which will be used to evaluate the evolution of the a.i.
        self.reward_window = []
# Here we initialize the nerual network and save it to the var model
        self.model = Network(input_size, nb_action)
# Here we initialize the memory and set the capacity
        self.memory = ReplayMemory(100000)                                       #Adjustable
# self.model.parameters is getting the params from the Network() class we create
# torch.optim.Adam(object, learnRate) is a class/func that takes the networks
# params and uses them to adjust the weight of the 'synapse', connection
# the second param is the learning rate. normally low so the ai has time to
# better explore and learn the full enviroment and all possibilities
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)         # Adjustable
# torch.Tensor() kind of like zip(). we need a list of tuples that we can batch
# all the differnt inputs into there own groups so they can assest.
# torch.Tensor() create this type of list and enters, in this case, 0
# [b1a,b1b,b1c] + [b2a,b2b,b2c] would be [(b1a,b2a),(b1b,b2b),(b1c,b2c)]
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
# here we create the last two var we need and asign them the val of 0
    # last refers to the action2rotation list in the map.py its an index
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
# torch.nn.functional(F) is a library that contains most of the torch nerual network
# function or actions. We use softmax which gives a probability rate to each action,
# that it will choose that action based on its Q value(choice success rate)
# self.model is an obj of th network class, we created in the init func above.
# The Network class takes the current state and returs the Q val of that state.
# the state param will be that state, it is already in the correct data format.
# still we will wrap in a torch Variable, now we will also add volatile = True,
# this is so the gradient rate will be included to all the graphs of the nn.Module,
# this will increase productivity and speed and save memory
# The T refers to the "temperature" var. It simply mulitplies the q val of the
# possible actions of that the current state. It increase how "sure" the A.I. is
        probs = F.softmax(self.model(Variable(state, volatile = True))*7) # T=100 Adjustable
# up to now what happen, probs = current state list of actions and Q values
# The next line picks one action, randomly base os the probability rate
        action = probs.multinomial(1)
# Since the action var is a torch.Tensor var we cant just return action, we have
# access its data and then index the data we want from the list(even if it only has 1 of each val)
        return action.data[0,0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
# here all params will be the output of samples, but in this instance we only
# want the action that was choosen in each event. gather() will do that for us.
# The 1 param will index the batch_state to get the action that was choosen
# we then take the batch_action, which right now is fake "vector" and more or less
# remove the fake data and replace it with the current data.
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
# batch_next_state is all the q val of the all the next possible states. We need
# the best Q val of the next possible states. We detach() the 'next_state' and
# use max(1)[0],1 indexs the actions, 0 indexs the q val of the state derived
# from those actions
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
# the gamma factor is multi by the max q val of all possible actions then added
# with the sum of the batch_rewards
        target = self.gamma*next_outputs + batch_reward
# smooth_l1_loss() is another torch.nn function that returns the Temperal Difference,
# the difference from the predicted output and the actual output, which is then
# back porpagated to adjust the "synapse" weight.  "Hover Loss"
        td_loss = F.smooth_l1_loss(outputs, target)
# As created in the init func the optimizer will back propagate and update the
# weights, but we must re-initialize it for every loop, that is what .zero_grad is
        self.optimizer.zero_grad()
# the backward() func is another torch func, this allows the optimizer to back propagate
        td_loss.backward(retain_graph = True)
# Here we are actual updating the weights with the step() func
        self.optimizer.step()

    def update(self, reward, new_signal):
# new_state is the compiled data of all the sensors, in this case sensor 1,2,3
# and orientaion and - orientaion( its pos and distance from goal).
# Like before we convert this data into a torch.Tensor and usure they are float int.
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
# here we use our push() func from earlier to update the memory. Note that
# self.last_state and new_state have previously been converted to a torch.Tensor
# so logically so does last_action and last_reward. last_reward is easy to do.
# since last action will be a single num we use torch.LongTensor to convert it
# we included the int() method to make sure we get an interger
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
# Here we use the select_action() func from above and apply the new_state which
# is the state the A.I. is in. This will select one of the avail options
        action = self.select_action(new_state)
# Here like before with the push function were setting a capacity for our batch
# size and delteing the fist element if it exceeds our limit
# below the first memory refers to the network class we init in the init func.
# the second memory refers to the memory attribute of the network class
        if len(self.memory.memory) > 100:                                       # Adujustable
# next we use our sample() collect a batch of memories and apply that val to these vars
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
# Now we use those random batches with the learn() to update"learn" the current
# states new weights and Q vals
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
# the next three line we are asigning the var we just created to our A.I. obj(brain)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
# Here like before with the push function were setting a capacity for our batch
# size and delteing the fist element if it exceeds our limit
        if len(self.reward_window) > 1000:                                       # Adujustable
            del self.reward_window[0]
# We must return the action. In the game this func is used to to update and
# since the select_action() is in this function it also makes the next play.
        return action

    def score(self):
# this will simply return the average of all the rewards in the reward_window +1
# the +1 is a safety to make sure the length is 0, which would cause a division
# error and crash you program
        return sum(self.reward_window)/(len(self.reward_window)+1.)

# The rest is self explanitory. just note that we use torch modules to save and load
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
