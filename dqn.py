import torch
import logging
import random
import numpy as np
from copy import deepcopy

from replaybuffer import UniformBuffer


class Dqn(torch.nn.Module):

    def __init__(self, valuenet, optimizer, buffersize):
        super().__init__()
        self.buffer = UniformBuffer(buffersize)
        self.network = valuenet
        self.targetnet = deepcopy(valuenet)
        self.opt = optimizer
        self.update_count = 0
        self.device = "cpu"

    def forward(self, state, epsilon=0.0):
        """
            Args:
                - state: Batch size 1 torch tensor.
                - epsilon: Random action probability.
        """
        self.eval()
        with torch.no_grad():
            values = self.network(state)
        values = values.squeeze()
        if len(values.shape) > 1:
            raise ValueError("Batch size of the given state tensor must be 1!")
        if random.uniform(0, 1) < epsilon:
            action = torch.randint(values.shape[-1], (1,)).long()
            value = values[action]
        else:
            action = torch.argmax(values)
            value = values[action]
        return action.item(), value.item()

    def td_loss(self, gamma, batchsize):

        batch = self.buffer.sample(batchsize)
        batch = self._batchtotorch(batch)

        with torch.no_grad():
            next_values = self.targetnet(batch.next_state)
            next_values = torch.max(next_values, dim=1, keepdim=True)[0]

        current_values = self.network(batch.state)
        current_values = current_values.gather(1, batch.action)

        target_value = next_values*(1 - batch.terminal)*gamma + batch.reward
        td_error = torch.nn.functional.smooth_l1_loss(current_values,
                                                      target_value)

        return td_error

    def update(self, gamma, batchsize, target_update_ratio, grad_clip=False):
        self.train()
        if self.update_count >= target_update_ratio:
            self.update_count = 0
            self.targetnet.load_state_dict(self.network.state_dict())

        self.opt.zero_grad()
        loss = self.td_loss(gamma, batchsize)
        loss.backward()

        if grad_clip:
            for param in self.network.parameters():
                param.grad.data.clamp_(-1, 1)

        self.opt.step()
        self.update_count += 1
        return loss.item()

    def push(self, state, action, reward, next_state, terminal):
        self.buffer.push(**dict(state=state,
                                action=action,
                                reward=reward,
                                next_state=next_state,
                                terminal=float(terminal)))

    def _batchtotorch(self, batch):
        state = self._totorch(batch.state, torch.float32)
        action = self._totorch(batch.action, torch.long).view(-1, 1)
        next_state = self._totorch(batch.next_state, torch.float32)
        terminal = self._totorch(batch.terminal, torch.float32).view(-1, 1)
        reward = self._totorch(batch.reward, torch.float32).view(-1, 1)
        return UniformBuffer.Transition(state, action,
                                        reward, next_state, terminal)

    def _totorch(self, container, dtype):
        tensor = torch.tensor(container).to(dtype)
        return tensor.to(self.device)

    def to(self, device):
        self.device = device
        super().to(device)
