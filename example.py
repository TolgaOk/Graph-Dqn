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
        epsilon = eps/args["episodes"]*epsilon_interval + args["terminal_eps"]

        while not done:
            state_torch = dqn._totorch(state, torch.float32).unsqueeze(0)
            act, value = dqn(state_torch, epsilon=epsilon)
            next_state, reward, done, _ = env.step(act)

            dqn.push(state, act, reward, next_state, done)
            if eps > args["update_begin"]:
                loss = dqn.update(args["gamma"],
                                  args["batchsize"],
                                  args["target_update_ratio"],
                                  grad_clip=True)
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

    for eps in range(args["episodes"]):
        train_reward, train_value, train_td = train(eps)
        if eps % 10 == 0:
            eval_reward, eval_value = evaluate()
        print("Episode: {}, train_reward:{}, eval reward: {}"
              .format(eps, train_reward, eval_reward))


def cartpole():
    args = dict(
        buffersize=100000,
        lr=0.001,
        target_update_ratio=100,
        gamma=0.99,
        episodes=1000,
        update_begin=75,
        init_eps=0.5,
        terminal_eps=0.0,
        batchsize=64
    )

    env = gym.make("CartPole-v1")
    in_size = env.observation_space.shape[0]
    n_act = env.action_space.n
    model = FullyConnectedModel(in_size, n_act)
    optim = torch.optim.Adam(model.parameters(), lr=args["lr"])
    dqn = Dqn(model, optim, args["buffersize"])

    learn(args, env, dqn)


def warehouse():
    pass

def warehouse_graph():
    pass

if __name__ == "__main__":
    cartpole()
