import gin
import os
import argparse
import utility
import torch
import pandas
import shutil
import numpy as np
from q_net import ParkingAgent 
from pfrl import agents, experiments, explorers
from pfrl import replay_buffers


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='../configs/default.gin')
parser.add_argument('--ckpt', type=str, default='./model/solar/best/')
args = parser.parse_args()

device = torch.device("cuda")

gin.parse_config_file(os.path.realpath(args.config))
hyper_params = utility.set_hyper_param()
if os.path.isdir("./logs/"):
     shutil.rmtree("./logs/")
os.mkdir("./logs/")

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

q_net = ParkingAgent(test_envs[0].action_space.n, test_envs[0].observation_space.shape[0])
optimizer = torch.optim.Adam(params = q_net.parameters(), lr = hyper_params["lr"])

replay_buffer = replay_buffers.PrioritizedReplayBuffer(
            hyper_params['mem_size'],
            alpha=0.6,
            beta0=0.4,
            betasteps=hyper_params['total_steps']/4,
            num_steps=1,
        )
explorer = explorers.LinearDecayEpsilonGreedy(
            0.,
            0.,
            hyper_params['eps_last_frame'],
            lambda: np.random.randint(test_envs[0].action_space.n),
        )

agent = agents.DoubleDQN(
            q_function=q_net,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            gamma=0.2,
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

agent.load(args.ckpt)
all_stats = []

for env in test_envs:
    eval_stats = experiments.eval_performance(env=env, 
                                              agent=agent, 
                                              n_steps=None, 
                                              n_episodes=1)

    stats = env.get_run_stats()
    all_stats.append(stats)
    env.render()

report = {}
for k in all_stats[0].keys():
    report[k] = []
for d in all_stats:
    for k in report.keys():
        report[k].append(d[k])

df = pandas.DataFrame(report)
df.to_csv("rl.csv")
