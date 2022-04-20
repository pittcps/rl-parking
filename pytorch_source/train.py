import gin
import os
import argparse
import utility
import torch
import numpy as np
from dqn import ParkingAgent 
from pfrl import agents, experiments, explorers
from pfrl import replay_buffers
from evaluation_callback import model_selection_callback

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='..\configs\default.gin')
parser.add_argument('--ckpt', type=str, default='')
args = parser.parse_args()

gin.parse_config_file(os.path.realpath(args.config))
hyper_params = utility.set_hyper_param()

if os.path.isdir("./model/") == False:
    os.mkdir("./model/")


if hyper_params['mode'] == 'solar':
    train_env, validation_envs, test_envs = utility.create_season_aware_environments(
                hyper_params['street_list'], 
                hyper_params['episode_max_steps'], 
                hyper_params['parking_data_dir'], 
                hyper_params['weather_data_dir'], 
                hyper_params['is_tensorflow']
        )
elif hyper_params['mode'] == 'cloud':
    train_env, validation_envs, test_envs = utility.create_environments(
                hyper_params['street_list'], 
                hyper_params['train_end'], 
                hyper_params['validation_end'], 
                hyper_params['episode_max_steps'], 
                hyper_params['parking_data_dir'], 
                hyper_params['is_tensorflow']
        )
else:
    print("Mode \"{}\" is not supported.".format(hyper_params['mode']))
    print("Please changen the to one of the following modes: [solar, cloud]")
    exit()

q_net = ParkingAgent(train_env.action_space.n, train_env.observation_space.shape[0])
optimizer = torch.optim.Adam(params = q_net.parameters(), lr = hyper_params["lr"])

replay_buffer = replay_buffers.PrioritizedReplayBuffer(
            hyper_params['mem_size'],
            alpha=0.6,
            beta0=0.4,
            betasteps=hyper_params['total_steps']/4,
            num_steps=1,
        )

explorer = explorers.LinearDecayEpsilonGreedy(
            1.0,
            0.1,
            hyper_params['eps_last_frame'],
            lambda: np.random.randint(train_env.action_space.n),
        )

agent = agents.DoubleDQN(q_function=q_net,
                         optimizer=optimizer,
                         replay_buffer=replay_buffer,
                         gamma=0.8,
                         explorer=explorer,
                         gpu=hyper_params['gpu_id'],
                         replay_start_size=hyper_params['mem_size'],
                         minibatch_size=hyper_params["batch_size"],
                         update_interval=4,
                         target_update_interval=200,
                         clip_delta=True,
                         n_times_update=1,
                         batch_accumulator="mean",
                         episodic_update_len=None,
                         )


if len(args.ckpt):
    print("loading weights from {}".format(args.ckpt))
    agent.load(args.ckpt)


eval_callback = model_selection_callback(validation_envs = validation_envs,
                                         start_episode = (hyper_params['train_end'] * (hyper_params['episodes']-10))/hyper_params['episode_max_steps'],
                                         filepath = "./model/best/")

experiments.train_agent(
            agent=agent,
            env=train_env,
            steps=hyper_params['total_steps'],
            outdir="./model/",
            max_episode_len = hyper_params['episode_max_steps'],
            step_hooks = [eval_callback],
            checkpoint_freq=100000,
            step_offset=0
            )

