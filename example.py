import torch
import numpy as np
import gym

from dqn import Dqn
from models import FullyConnectedModel


def learn(args, env, dqn):

    def train(eps):
        state = env.reset()
        done = False
        dqn.train()
        eps_reward = 0
        eps_value = []
        eps_loss = []

        epsilon_interval = args["init_eps"] - args["terminal_eps"]
        epsilon = (1 - eps/args["episodes"])*epsilon_interval +\
            args["terminal_eps"]
        while not done:
            state_torch = dqn._totorch(state, torch.float32).unsqueeze(0)
            act, value = dqn(state_torch, epsilon=epsilon)
            next_state, reward, done, _ = env.step(act)

            dqn.push(state, act, reward, next_state, done)
            if eps > args["update_begin"]:
                loss = dqn.update(args["gamma"],
                                  args["batchsize"],
                                  args["target_update_ratio"],
                                  grad_clip=False)
                eps_loss.append(loss)
            eps_reward += reward
            eps_value.append(value)
            state = next_state

        return eps_reward, np.mean(eps_value), np.mean(eps_loss)

    def evaluate():
        state = env.reset()
        done = False
        dqn.eval()
        eps_reward = 0
        eps_value = []

        while not done:
            state_torch = dqn._totorch(state, torch.float32).unsqueeze(0)
            act, value = dqn(state_torch)
            state, reward, done, _ = env.step(act)

            eps_reward += reward
            eps_value.append(value)

        return eps_reward, np.mean(eps_value)

    train_rewards = []
    eval_rewards = []
    td_losses = []
    for eps in range(args["episodes"]):
        train_reward, train_value, train_td = train(eps)
        if eps % 10 == 0:
            eval_reward, eval_value = evaluate()
        train_rewards.append(train_reward)
        eval_rewards.append(eval_reward)
        print("Episode: {}, train_reward:{}, eval reward: {}, td_loss: {}"
              .format(eps,
                      np.mean(train_rewards[-100:]),
                      np.mean(eval_rewards[-100:]),
                      train_td.item(), '.2f'))


def cartpole():
    args = dict(
        buffersize=100000,
        lr=0.001,
        target_update_ratio=200,
        gamma=0.99,
        episodes=1000,
        update_begin=78,
        init_eps=0.7,
        terminal_eps=0.1,
        batchsize=32
    )

    env = gym.make("CartPole-v1")
    in_size = env.observation_space.shape[0]
    n_act = env.action_space.n
    model = FullyConnectedModel(in_size, n_act)
    optim = torch.optim.Adam(model.parameters(), lr=args["lr"])
    dqn = Dqn(model, optim, args["buffersize"])

    learn(args, env, dqn)


def warehouse():
    from gymcolab.envs.warehouse import Warehouse
    from models import VanillaDqnModel

    args = dict(
        buffersize=100000,
        lr=0.001,
        target_update_ratio=100,
        gamma=0.99,
        episodes=10000,
        update_begin=75,
        init_eps=0.5,
        terminal_eps=0.0,
        batchsize=32,
    )
    device = "cuda"

    env = Warehouse()
    n_nodes, height, width = env.observation_space.shape
    n_act = env.action_space.n - 1
    assert height == width, "Environment map must be square"
    model = VanillaDqnModel(n_nodes, height, n_act)
    optim = torch.optim.Adam(model.parameters(), lr=args["lr"])
    dqn = Dqn(model, optim, args["buffersize"])
    dqn.to(device)
    learn(args, env, dqn)


def warehouse_graph():
    from gymcolab.envs.warehouse import Warehouse
    from models import GraphDqnModel

    args = dict(
        buffersize=100000,
        lr=0.0001,
        target_update_ratio=100,
        gamma=0.99,
        episodes=10000,
        update_begin=75,
        init_eps=0.9,
        terminal_eps=0.1,
        batchsize=32,
    )
    device = "cuda"

    env = Warehouse()
    n_nodes, height, width = env.observation_space.shape
    n_act = env.action_space.n
    assert height == width, "Environment map must be square"
    adj = torch.tensor([
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 1, 0, 0]]
    ]).float().to(device)
    n_edges = 3

    class GraphDqnModelAdj(GraphDqnModel):

        def __init__(self, n_edges, n_nodes, mapsize, n_act, adj):
            super().__init__(n_edges, n_nodes, mapsize, n_act)
            self.adj = adj

        def forward(self, objmap):
            return super().forward(self.adj, objmap)

    model = GraphDqnModelAdj(n_edges, n_nodes, height, n_act, adj)
    optim = torch.optim.Adam(model.parameters(), lr=args["lr"])
    dqn = Dqn(model, optim, args["buffersize"])
    dqn.to(device)
    learn(args, env, dqn)


if __name__ == "__main__":
    cartpole()
