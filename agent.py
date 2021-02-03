from random import random, randint
import torch

from model import DQN
from replayMemory import ReplayMemory
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    def __init__(self, args, n_actions):
        self.model = DQN(args.img_width, args.img_height, args.channels, n_actions).to(device)
        self.n_action = n_actions
        self.epsilon_start = args.epsilon
        self.epsilon = args.epsilon
        self.decay_start = args.decay_start
        self.decay_end = args.n_epochs * 0.8
        self.memory = ReplayMemory(args)
        self.batch_size = args.batch_size
        self.actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def update_esplison(self, epoch):
        if epoch < self.decay_end:
            ep_step = (self.epsilon_start - 0.1) / self.decay_end

            self.epsilon = self.epsilon_start - (ep_step * epoch)
        else:
            self.epsilon = 0

    def act(self, state):
        if random() <= self.epsilon:
            return randint(0, self.n_action - 1)
        else:
            return self.get_best_action(state)

    def get_best_action(self, state):
        q = self.get_q_values(state)
        m, index = torch.max(q, 1)
        action = index.item()

        return action

    def get_q_values(self, state):
        state = torch.from_numpy(state).to(device)
        return self.model(state)

    def get_best_action_wGrad(self, state):
        state = torch.from_numpy(state).to(device)
        
        
        ### CODE AJOUTE ###
        # Track gradient on input vars but not model
        state.requires_grad = True
        for param in self.model.parameters():
            param.requires_grad = False
        ### ~ ###
        
        
        q, latent = self.model(state)

        m, index = torch.max(q, 1)
        action = index.item()
        
        
        ### CODE AJOUTE ###
        # Create one-hot target
        target = torch.zeros(q.shape)
        target[:, index] = 1
        
        # Compute gradient
        q.backward(gradient=target)
        state.detach_()

        # Get gradient and detach it from the graph
        grads = state.grad[0, 0, :, :].numpy()
        
        # Keep only positive values
        grads[grads < 0] = 0
        
        # Normalize
        grads = grads / np.max(grads)
        ### ~ ###

        
        grads *= 255
        grads = grads.astype(np.int8)

        return action, grads.transpose((1, 0)), latent
