#!/usr/bin/env python3

import gym
import json
import numpy
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import datetime
import os

import time
from gym import wrappers
# from stable_baselines3.common.monitor import Monitor
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.head = nn.Linear(32, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return self.head(x)


def select_action(state, eps_start, eps_end, eps_decay):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(0)[1].view(1, 1), eps_threshold
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), eps_threshold


def optimize_model(batch_size, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, torch.squeeze(action_batch, 2))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# import our training environment
if __name__ == '__main__':

    rospy.init_node('turtlebot3_world_qlearn', anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot3/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_turtlebot3_openai_example')
    # outdir = pkg_path + '/training_results'
    #env = wrappers.Monitor(env, outdir, force=True)
    #env = Monitor(env, outdir, allow_early_resets=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    gamma = rospy.get_param("/turtlebot3/gamma")
    epsilon_start = rospy.get_param("/turtlebot3/epsilon_start")
    epsilon_end = rospy.get_param("/turtlebot3/epsilon_end")
    epsilon_decay = rospy.get_param("/turtlebot3/epsilon_decay")
    n_episodes = rospy.get_param("/turtlebot3/n_episodes")
    batch_size = rospy.get_param("/turtlebot3/batch_size")
    target_update = rospy.get_param("/turtlebot3/target_update")

    running_step = rospy.get_param("/turtlebot3/running_step")

    # Initialises the algorithm that we are going to use for learning
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    n_observations = 360 #PARAMETRO CHE DEVE ESSERE UGUALE A QUELLO IN /src/openai_ros/openai_ros/src/openai_ros/task_envs/turtlebot3/config/turtlebot3_world.yaml

    # initialize networks with input and output sizes
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)
    episode_durations = []
    steps_done = 0

    start_time = time.time()
    highest_reward = 0
    
    reward_for_episode = []
    
    # Starts the main training loop: the one about the episodes to do
    for i_episode in range(n_episodes):
        rospy.logdebug("############### START EPISODE=>" + str(i_episode))

        cumulated_reward = 0
        
        
        
        done = False

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        
        # resize observations to pass it to the net
        #observation = numpy.array(observation)
        #observation.resize((1,n_observations), refcheck=False)
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print()
        print()
        print()
        #print(observation.shape)
        print()
        print()
        print()
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        
        state = torch.tensor(observation, device=device, dtype=torch.float)
        #state = ''.join(map(str, observation))

        for t in count():
            rospy.logwarn("############### Start Step=>" + str(t))
            # Select and perform an action
            action, epsilon = select_action(state, epsilon_start, epsilon_end, epsilon_decay)
            rospy.logdebug("Next action is:%d", action)

            observation, reward, done, info = env.step(action.item())
            rospy.logdebug(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            reward = torch.tensor([reward], device=device)
            
            #next_state = ''.join(map(str, observation))
            next_state = torch.tensor(observation, device=device, dtype=torch.float)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Perform one step of the optimization (on the policy network)
            rospy.logdebug("# state we were=>" + str(state))
            rospy.logdebug("# action that we took=>" + str(action))
            rospy.logdebug("# reward that action gave=>" + str(reward))
            rospy.logdebug("# episode cumulated_reward=>" + str(cumulated_reward))
            rospy.logdebug("# State in which we will start next step=>" + str(next_state))
            optimize_model(batch_size, gamma)
            if done:
                episode_durations.append(t + 1)
                rospy.logdebug("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(t + 1)])
                break
            else:
                rospy.logdebug("NOT DONE")
                state = next_state

            rospy.logwarn("############### END Step=>" + str(t))
            # Update the target network, copying all weights and biases in DQN
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(i_episode + 1) + " - gamma: " + str(
            round(gamma, 2)) + " - epsilon: " + str(round(epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))
        
        reward_for_episode.append(cumulated_reward)

    rospy.loginfo(("\n|" + str(n_episodes) + "|" + str(gamma) + "|" + str(epsilon_start) + "*" +
                   str(epsilon_decay) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()
    
    min_rew = min(reward_for_episode)
    reward_for_episode = [x - min_rew for x in reward_for_episode]

    timestamp = str(datetime.datetime.now()).replace(' ', '_')
    outdir = f"../simulation_ws/training_results/results-{timestamp}"
    os.mkdir(outdir)
    with open(f"{outdir}/results-{timestamp}.json", "w") as f:
        dictionary = {"time": last_time_steps.tolist(), "rewards": reward_for_episode, "model": str(policy_net), 
                      "gamma": gamma, "epsilon_start":epsilon_start, "epsilon_end":epsilon_end, "epsilon_decay":epsilon_decay,
                      "n_episodes":n_episodes, "batch_size":batch_size, "optimizer": optimizer}
        json.dump(dictionary, f)
    

    unit = round(len(reward_for_episode)/20)
    x_values = [i*unit for i in range(1,21)]

    fig, ax = plt.subplots(3)
    ax[0].plot(x_values, reward_for_episode, color="blue")
    ax[0].set_title("Reward for episode")

    ax[1].plot(x_values, last_time_steps, color="orange")
    ax[1].set_title("Last time per episode")

    ax[2].plot(x_values, reward_for_episode, color="blue")
    ax[2].plot(x_values, last_time_steps, color="orange")
    ax[2].set_title("Combined plots")

    plt.setp(ax, xticks=x_values)
    plt.tight_layout()

    plt.show(block=False)
    plt.savefig(f"{outdir}/plot.png")
    
    # print("Parameters: a="+str)
    # rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    # rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
