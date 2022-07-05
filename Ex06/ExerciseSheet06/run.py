import os

import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

import models
import planners
import simple_control_env


parser = argparse.ArgumentParser()
parser.add_argument("--lunarlander", action="store_true", help="Use LunarLander environment")
parser.add_argument("--train", action="store_true", help="Training vs evaluation")
parser.add_argument("--cem", action="store_true", help="Use CEM in simple control environment")
parser.add_argument("--render", action="store_true", help="Render the environment")
parser.add_argument("--record", action="store_true", help="Save images of environment to hard disk")
parser.add_argument("--seed", type=int, default=0, help="Seed for RNG")

args = parser.parse_args()

use_simple_environment = not args.lunarlander
use_CEM = args.cem
train = args.train
render = args.render
record = args.record
os.makedirs("images", exist_ok=True)
seed = args.seed

assert(not(use_simple_environment and train))

model_id = "test"
state_path = None if train else "models/test/2999.pth"


torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# Make environment a bit easier to solve
import gym.envs.box2d.lunar_lander as lunar_lander
lunar_lander.INITIAL_RANDOM = 500  # Do not change this


# Initialize environment
if use_simple_environment:
    env = simple_control_env.SimpleControlGym()
else:
    env = gym.make("LunarLander-v2", continuous=True, enable_wind=False)
action_size = env.action_space.shape[0]
observation_size = env.observation_space.shape[0] - 2  # Ignore leg contact


# Initialize model
if use_simple_environment:
    model = models.SimpleModel(env)
else:
    os.makedirs(os.path.join("models", model_id), exist_ok=True)
    input_size = action_size + observation_size
    hidden_size = 256
    output_size = observation_size
    model = models.NeuralNetworkModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
    )


# Define criterion for the CEM planner
def criterion_simple_environment(observation):
    target_pos = torch.Tensor([0.5, 0.5]).repeat(observation.shape[0], observation.shape[1], 1)
    loss_pos = torch.nn.functional.mse_loss(observation, target_pos)
    return loss_pos

def criterion_lunar_lander(observation):
    # Adapt this if necessary
    target_pos_x = torch.Tensor([0]).repeat(observation.shape[0], observation.shape[1], 1)
    target_pos_y = torch.Tensor([0]).repeat(observation.shape[0], observation.shape[1], 1)
    target_ang = torch.Tensor([0]).repeat(observation.shape[0], observation.shape[1], 1)

    loss_pos_x = 2.4 * torch.nn.functional.mse_loss(observation[..., 0:1], target_pos_x)
    loss_pos_y = 0.2 * torch.nn.functional.mse_loss(observation[..., 1:2], target_pos_y)
    loss_ang = 1.4 * torch.nn.functional.mse_loss(observation[..., 4:5], target_ang)

    target_vel_y = torch.Tensor([-0.1]).repeat(observation.shape[0], observation.shape[1], 1)
    loss_vel_y = torch.nn.functional.mse_loss(observation[..., 3:4], target_vel_y, reduction="none").squeeze()
    factor_vel_y = torch.zeros(target_pos_y.shape[0])
    factor_vel_y[observation[..., 1:2].squeeze() < 0.5] = 0.3
    factor_vel_y[observation[..., 1:2].squeeze() < 0.2] = 0.5
    loss_vel_y = (factor_vel_y * loss_vel_y).mean()

    loss = loss_pos_x + loss_pos_y + loss_vel_y + loss_ang
    return loss

if use_simple_environment:
    criterion = criterion_simple_environment
else:
    criterion = criterion_lunar_lander


# Initialize planner
horizon = 50
num_inference_cycles = 3
num_predictions = 50
num_elites = 5
num_keep_elites = 2

planner_random = planners.RandomPlanner(
    action_size=action_size,
    horizon=horizon
)
planner_cem = planners.CrossEntropyMethod(
    action_size=action_size,
    horizon=horizon,
    num_inference_cycles=num_inference_cycles,
    num_predictions=num_predictions,
    num_elites=num_elites,
    num_keep_elites=num_keep_elites,
    criterion=criterion,
    policy_handler=lambda x: x.clamp(env.action_space.low[0], env.action_space.high[0]),
    var=1,
)


# Initialize optimizer
if train:
    # TODO: define optimizer
    optimizer = None


# Adapt if necessary
epochs = 3000
sequence_length = 200


epoch = 0
losses = []

# Load pretrained model and losses if state_path is defined
if not use_simple_environment and state_path:
    state = torch.load(state_path)
    model.load_state_dict(state["model_state"])
    if train:
        optimizer.load_state_dict(state["optimizer_state"])
    epoch = state["epoch"]
    losses = state["losses"]


# Run
while epoch < epochs:
    observation_old = env.reset(seed=epoch+seed)
    observation_old = torch.Tensor(observation_old)
    loss = 0
    done = False
    counter = 0
    heuristic = torch.rand([1]).item() < 0.5
    while not done and counter <= sequence_length:
        if train:
            # To generate training data, we either use the heuristic given by
            # the environment or sample uniformly
            if heuristic:
                actions = gym.envs.box2d.lunar_lander.heuristic(env, observation_old)
            else:
                actions = env.action_space.sample()
            actions = torch.Tensor(actions).unsqueeze(dim=0)

            # Randomly switch between heuristic and random actions to create
            # diverse data
            if random.random() < 0.02:
                heuristic = not heuristic
        else:
            if use_simple_environment:
                if use_CEM:
                    actions, _ = planner_cem(model, observation_old)
                else:
                    actions, _ = planner_random(model, observation_old)
            else:
                # Ignore whether the legs touch the ground during planning
                actions, _ = planner_cem(model, observation_old[..., :-2])  # :-2 to Ignore leg contact

        action = actions[0]
        if train:
            inp = torch.cat([observation_old[..., :-2], torch.Tensor(action)])  # :-2 to Ignore leg contact
            prediction = model.forward(inp)
        observation, _, done, _ = env.step(action.numpy())
        observation = torch.Tensor(observation)
        if render:
            env.render()
        if record:
            img = env.render(mode="rgb_array")
            plt.imshow(img)
            plt.savefig(os.path.join("images", f"env_{counter:05d}.png"))
            plt.close()
        if train and counter % 50 == 49:
            label = observation
            loss += torch.nn.functional.mse_loss(prediction, label[..., :-2])  # :-2 to Ignore leg contact
            loss /= 50
            print(f"{epoch:03d}: ", loss)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = 0
        observation_old = observation.clone()
        counter += 1
    if train:
        state_dict = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "losses": losses,
        }
        torch.save(
            state_dict,
            os.path.join("models", model_id, f"{epoch:04d}.pth")
        )
    epoch += 1
    #print()


# Create plots of the loss with log scale
if train:
    plt.plot(losses)
    plt.yscale("log")
    plt.savefig(f"loss_{model_id}.png")
    plt.close()
    # Plot running average of loss
    plt.plot(np.convolve(np.array(losses), np.ones(50)/50))
    plt.yscale("log")
    plt.savefig(f"loss_{model_id}_convolved.png")
    plt.close()
