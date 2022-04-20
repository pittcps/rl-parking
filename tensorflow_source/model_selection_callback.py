
import rl.callbacks 
from customgym import parking_env

class model_selection_callback(rl.callbacks.Callback):

    def __init__(self, validation_model, validation_envs, selection_start_episode, filepath):
        super(model_selection_callback, self).__init__()
        
        self.validation_model = validation_model
        self.validation_envs = validation_envs
        self.file_path = filepath
        self.selection_start_episode = selection_start_episode
        
        self.best_reward = -999999999

    def on_episode_end(self, episode, logs={}):

        if episode >= self.selection_start_episode:
            print("Calculating validation results")
            sum_reward = 0
            for i in range(len(self.validation_envs)):
                self.validation_envs[i].reset()
                weights = self.model.model.get_weights()
                self.validation_model.model.set_weights(weights)
                self.validation_model.test(self.validation_envs[i], nb_episodes=1, visualize=False)
                sum_reward += self.validation_envs[i].total_reward

            mean_reward = sum_reward / len(self.validation_envs)
            
            if mean_reward > self.best_reward:

                self.best_reward = mean_reward
                self.model.save_weights(self.file_path, overwrite=True)

                print("Better model found, reward: ", self.best_reward)
            
