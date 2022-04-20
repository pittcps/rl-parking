from pfrl import experiments

class model_selection_callback():

    def __init__(self, validation_envs, start_episode, filepath):
        self.validation_envs = validation_envs
        self.file_path = filepath
        
        self.best_reward = -999999999
        self.best_step = -1
        self.last_seen_episode = 0
        self.start_episode = start_episode

    def __call__(self, env, agent, step):

        if self.last_seen_episode != env.episode_num:
            self.last_seen_episode = env.episode_num
            if env.episode_num < self.start_episode:
                return
            print("Calculating validation results")
            sum_reward = 0
            for i in range(len(self.validation_envs)):
                self.validation_envs[i].reset()

                experiments.eval_performance(env=self.validation_envs[i], agent=agent, n_steps=None, n_episodes=1)

                if self.validation_envs[i].current_step < len(self.validation_envs[i].df_list[0]) - 1:
                    # unfinished

                    sum_reward = -9999999999
                    break

                sum_reward += self.validation_envs[i].total_reward


            mean_reward = sum_reward / len(self.validation_envs)
            
            if mean_reward > self.best_reward:

                self.best_reward = mean_reward
                self.best_step = step

                agent.save(dirname=self.file_path)

                print("Better model found at step {}, reward: {}: ".format(step, self.best_reward))
            else:

                print("did not find a better model at step {}".format(step))
