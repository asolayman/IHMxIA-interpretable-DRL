import argparse

import torch
from torch import optim
from tqdm import trange
from agent import DQNAgent
from env import DoomEnv
import numpy as np
from time import time
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parsearg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--scenario', type=str, default="basic", help='vizdoom scenario')
    parser.add_argument('--test_mod', type=bool, default=False, help=' test or train model')

    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
    parser.add_argument('--learning_steps_per_epoch', type=int, default=2000, help='number of steps per epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.00025, help='adam: learning rate')
    parser.add_argument('--discount_factor', type=float, default=0.9, help='adam: learning rate')
    parser.add_argument('--epsilon', type=float, default=0.8, help='exploration rate')
    parser.add_argument('--decay_start', type=int, default=1, help='exploration rate decay start')
    parser.add_argument('--memory_size', type=int, default=10000, help='size of the memory replay')

    parser.add_argument('--channels', type=int, default=
    1, help='channels of image')
    parser.add_argument('--img_height', type=int, default=64, help='size of image height')
    parser.add_argument('--img_width', type=int, default=112, help='size of image width')
    parser.add_argument('--frame_repeat', type=int, default=6, help='frame skip')

    parser.add_argument('--model_savefile', type=str, default="model.pth", help='frame skip')

    # parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    # parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    # parser.add_argument('--checkpoint_interval', type=int, default=50, help='interval between model checkpoints')

    return parser.parse_args()


def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if agent.memory.size > args.batch_size:
        s1, a, s2, isterminal, r = agent.memory.get_sample(args.batch_size)

        q = agent.get_q_values(s2).cpu().data.numpy()
        q2 = np.max(q, axis=1)

        target_q = agent.get_q_values(s1).cpu().data.numpy()
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r

        target_q[np.arange(target_q.shape[0]), a] = r + args.discount_factor * (1 - isterminal) * q2
        learn(s1, target_q)


def learn(s1, target_q):
    s1 = torch.from_numpy(s1).to(device)
    target_q = torch.from_numpy(target_q).to(device)
    output = agent.model(s1)
    loss = criterion(output, target_q).to(device)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def train(args):
    print("Starting the training!")
    time_start = time()
    for epoch in range(args.n_epochs):
        print("\nEpoch %d\n-------" % (epoch + 1))
        train_episodes_finished = 0
        train_scores = []
        agent.update_esplison(epoch)
        print(agent.epsilon)

        print("Training...")
        env.game.new_episode()
        left_count = 0
        for learning_step in trange(args.learning_steps_per_epoch, leave=False):
            state1 = env.get_state()
            action = agent.act(state1)

            if action == 0:
                left_count += 1

            reward = env.game.make_action(agent.actions[action], args.frame_repeat)
            state2 = None
            if not env.game.is_episode_finished():
                state2 = env.get_state()

            agent.memory.add_transition(state1, action, state2, env.game.is_episode_finished(), reward)

            if env.game.is_episode_finished():
                score = env.game.get_total_reward()
                train_scores.append(score)
                env.game.new_episode()
                train_episodes_finished += 1
            learn_from_memory()
        print(left_count)

        print("%d training episodes played." % train_episodes_finished)

        train_scores = np.array(train_scores)

        print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
              "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

    print("Saving the network weigths to:", args.model_savefile)
    torch.save(agent.model, args.model_savefile)

    print("\nTesting...")
    test_scores = []
    # args = args.copy()
    # args["test_mod"] = True
    input(" Next \n")
    new_env = DoomEnv(args, True)
    for test_episode in trange(10, leave=False):
        new_env.game.new_episode()
        while not new_env.game.is_episode_finished():
            state = new_env.get_state()
            best_action_index = agent.get_best_action(state)
            # best_action_index = 3 #agent.get_best_action(state)

            # new_env.game.make_action([1,0,0], args.frame_repeat)
            new_env.game.make_action(agent.actions[best_action_index], args.frame_repeat)
        r = new_env.game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print("Results: mean: %.1f +/- %.1f," % (
        test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
          "max: %.1f" % test_scores.max())

    print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    env.game.close()
    print("======================================")
    print("Training finished!")


if __name__ == '__main__':
    args = parsearg()
    env = DoomEnv(args, False)
    criterion = nn.MSELoss()
    agent = DQNAgent(args, env.n_actions)
    optimizer = optim.RMSprop(agent.model.parameters(), lr=args.lr)
    train(args)
