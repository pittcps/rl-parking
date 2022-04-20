import utility
import q_net
import os
import argparse
import gin
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import PrioritizedMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from __future__ import division
from model_selection_callback import model_selection_callback


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='street_parking-v0')
parser.add_argument('--ckpt', type=str, default=None)
args = parser.parse_args()

gin.parse_config_file(os.path.realpath(args.config))
hyper_params = utility.set_hyper_param()


def init_dqn(env, model):
    memory = PrioritizedMemory(
        limit=hyper_params['mem_size'], 
        alpha=.6, 
        start_beta=.4, 
        end_beta=1., 
        steps_annealed=hyper_params['total_steps'], 
        window_length=1
        )

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), 
        attr='eps', 
        value_max=1., 
        value_min=.1, 
        value_test=0.,
        nb_steps=hyper_params['total_steps']
    )
    dqn = DQNAgent(
        model=model, 
        nb_actions=env.action_space.n, 
        policy=policy, 
        memory=memory,
        enable_double_dqn=True, 
        enable_dueling_network=True, 
        gamma=.2, 
        target_model_update=200,
        train_interval=4, 
        delta_clip=1., 
        nb_steps_warmup=hyper_params['mem_size'])
    dqn.compile(Adam(lr=hyper_params['lr']), metrics=['mse'])
    return dqn

def test_model(dqn, envs):
        
    sum_reward = 0
    sum_accuracy = 0
    sum_reduction = 0

    for i in range(len(envs)):     
        dqn.test(envs[i], nb_episodes=1, visualize=False)

        sum_reward += envs[i].total_reward
        sum_accuracy += envs[i].total_seen_highs/envs[i].total_highs
        sum_reduction += envs[i].total_off_time/(envs[i].total_off_time + envs[i].total_on_time)

    print("Mean reward: ", sum_reward/len(envs))
    print("Mean accuracy: ", sum_accuracy/len(envs))
    print("Mean reduction: ", sum_reduction/len(envs))

    return sum_reward/len(envs)

if __name__ == "__main__":

    train_env, validation_envs, test_envs = train_env, validation_envs, test_envs = utility.create_environments(
            hyper_params['street_list'], 
            hyper_params['train_end'], 
            hyper_params['validation_end'], 
            hyper_params['episode_max_steps'], 
            hyper_params['parking_data_dir'], 
            hyper_params['is_tensorflow']
    )

    model = q_net.get_model(train_env)
    dqn = init_dqn(train_env, model)
        

    if args.mode == 'train':
        validation_model = init_dqn(validation_envs[0], q_net.get_model(train_env))

        weights_filename = 'model/dqn_{}_weights.h5f'.format(args.env_name)
        best_model_path = 'model/dqn_{}_best_weights.h5f'.format(args.env_name)
        checkpoint_weights_filename = 'model/dqn_' + args.env_name + '_weights_{step}.h5f'
        log_filename = 'logs/dqn_{}_log.json'.format(args.env_name)

        model_selection_recall = model_selection_callback(
            validation_model = validation_model, 
            validation_envs = validation_envs, 
            selection_start_episode = (hyper_params['train_end'] * (hyper_params['episodes']-10))/hyper_params['episode_max_steps'], 
            filepath = best_model_path)
        
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=hyper_params['train_end'])]
        callbacks += [FileLogger(log_filename, interval=100000)]
        callbacks += [model_selection_recall]

        dqn.fit(
            train_env, 
            callbacks=callbacks, 
            nb_steps=hyper_params['total_steps'], 
            verbose=2, 
            nb_max_episode_steps=hyper_params['episode_max_steps']
        )

        dqn.save_weights(weights_filename, overwrite=True)

        dqn.load_weights(best_model_path)

        test_model(dqn, test_envs)


    elif args.mode == 'test':
        best_model_path = 'model/dqn_{}_best_weights.h5f'.format(args.env_name)
        if args.ckpt:
            best_model_path = args.ckpt
        dqn.load_weights(best_model_path)
        test_model(dqn, test_envs)