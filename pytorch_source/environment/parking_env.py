import gym
import random
import math
import random
import logging
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class VanillaStreetParkingEnv(gym.Env):

    def __init__(self):
        super(VanillaStreetParkingEnv, self).__init__()
        self.current_camera_state = True
        self.total_reward = 0

        self.history_len = 60 + 1
        self.history_interval_len = 10
        self.history = []
        self.init_history()

        self.gpu_power = 250
        # we run inference 5 time per second
        self.inference_cost = (((self.gpu_power / 60) / 60) / 5) * 60
        self.missed_high_cost = 50.
        # Camera power usage in watts for one minute
        self.power_usage = 40 / 60
        self.episode_rewards = []

        self.norm_power_usage, self.norm_missed_high_cost, self.norm_inference_cost = self.normalize(self.power_usage,
                                                                                                    self.missed_high_cost,
                                                                                                    self.inference_cost)
        # Action space consist of one value:
        # 0 means off and 1 means on
        self.action_space = gym.spaces.Discrete(2)

        # Observation contains:
        # day_of_week[0], day_of_week[1]: Cyclic represnetation of day of the week. Each value is in [-1., 1.]
        # time[0], time[1]: Cyclic represnetation of time of the day. Each value is in [-1., 1.]
        # history[i][0], history[i][1] for i in [0, 5]: history[i][0] is 0 or 1 showing whether the percentage is known or not. history[i][1] is the occupancy percentage [0., 1.] and it's 0 for unknown values.
        # is_known: Whether curent percentage is known or not. It's 0 or 1
        # percent: Occupancy percent. In [0., 1.]
        # camera_state[0], camera_state[1]: shows if camera is on or off after the end of current action. If on camera_state[0] is 1 and camera_state[1] is 0, other wise they are 0 and 1 respectively. 

        lows = np.array([-1, -1, #day_of_week
                         -1, -1, #time
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, #history
                         0, #is_known
                         0, #percent
                         0, 0#camera_state
                        ])
        highs = np.array([1, 1, #day_of_week
                         1, 1, #time
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, #history
                         1, #is_known
                         1, #percent
                         1, 0#camera_state
                        ])
        self.observation_space = gym.spaces.Box(low=lows, high=highs, dtype=np.float32)

    def my_init(self, df_list, debug=False, interval_len=1, mode=0, training_max_steps= 7 * 24 * 60, streets=[], is_tensorflow = False):

        self.MAX_STEPS = training_max_steps
        self.df_list = df_list
        self.debug = debug
        self.interval_len = interval_len
        self.mode = mode
        self.df_index = 0
        self.streets = streets
        self.episode_num = 0
        self.is_tensorflow = is_tensorflow

        if mode == 0:
            self.writer = SummaryWriter()
        # create file handler which logs even debug messages

        if mode == 2:
            path = 'logs/parking_multi_street.csv'
            logger_name = 'multifile'
            if len(streets) == 1:
                logger_name = str(streets[0])
                path = 'logs/parking_{}.csv'.format(streets[0])

            self.test_logger = logging.getLogger(logger_name)
            self.test_logger.setLevel(logging.INFO)
            fh = logging.FileHandler(path)
            fh.setLevel(logging.INFO)
            # add the handlers to the logger
            self.test_logger.addHandler(fh)

        for i in range(len(self.df_list)): 
            self.df_list[i] = self.df_list[i].reset_index(drop=True)

    def init_history(self):
        for i in range(0, self.history_len):
            self.history.append([0, 0])

    def add_to_history(self, is_known, precentage):
        self.history.pop(0)
        self.history.append([is_known, precentage])

        if len(self.history) != self.history_len:
            print("Wrong history length")

    def get_history(self):
        hist = []
        sum = 0
        cnt = 0

        for i in range(0, self.history_len - 1):

            # Precentage is known
            if self.history[i][0] == 1:
                cnt += 1
                sum += self.history[i][1]

            if (i + 1) % 10 == 0:
                if cnt > 0:
                    hist.append([1, sum / cnt])
                else:
                    hist.append([0, 0])

                sum = 0
                cnt = 0

        return hist

    def sin_cos(self, v):
        t = 2. * math.pi * v
        return (math.sin(t), math.cos(t))

    def _next_observation(self):

        total_passed_minutes = int(self.df_list[self.df_index].loc[self.current_step, 'Hour']) * 60 + int(
            self.df_list[self.df_index].loc[self.current_step, 'Minute'])
        camera_state = [1, 0]

        if self.current_camera_state:
            camera_state = [0, 1]

        is_known = 0
        precent = 0

        if self.current_camera_state == True:
            is_known = 1
            precent = self.df_list[self.df_index].loc[self.current_step, 'Percent']

        day_of_week = self.sin_cos(int(self.df_list[self.df_index].loc[self.current_step, 'Day of week']) / 6)
        time = self.sin_cos(total_passed_minutes / 1440)

        self.add_to_history(is_known, precent)
        hist = self.get_history()

        obs = (day_of_week[0], day_of_week[1],
         time[0], time[1],
         hist[0][0], hist[0][1],
         hist[1][0], hist[1][1],
         hist[2][0], hist[2][1],
         hist[3][0], hist[3][1],
         hist[4][0], hist[4][1],
         hist[5][0], hist[5][1],
         is_known, precent,
         camera_state[0], camera_state[1])

        if self.is_tensorflow == False:
            obs = [torch.tensor(x, dtype=torch.float) for x in obs]
            obs = torch.stack(obs, dim=0)
        return obs

    def _take_action(self, action):
        action += 1

        missed_high = 0
        seen_high = 0
        missed_medium = 0
        on_length = 0
        off_length = 0
        need_startup = 0

        if action == 2:

            if self.current_camera_state == False:
                need_startup = 1

            self.current_camera_state = True
            on_length = self.interval_len

            index = self.current_step + 1
            label = self.df_list[self.df_index].loc[index, 'Label']
            self.total_highs += (label == 2)

            if label == 2:
                seen_high += 1

        else:
            self.current_camera_state = False
            off_length = self.interval_len

            index = self.current_step + 1
            label = self.df_list[self.df_index].loc[index, 'Label']
            self.total_highs += (label == 2)

            if label == 2:
                missed_high += 1
            elif label == 1:
                missed_medium += 1

        if on_length > 0 and seen_high == 0:
            self.total_useless_on_time += on_length

        self.total_time += self.interval_len
        self.current_step += 1
        self.total_off_time += off_length
        self.total_on_time += on_length
        self.total_startups += need_startup
        self.total_missed_highs += missed_high
        self.total_missed_mediums += missed_medium
        self.total_seen_highs += seen_high

        return need_startup, missed_high, missed_medium, seen_high, on_length, off_length

    def normalize(self, power_usage, missed_high_cost, inference_cost):
        tot = inference_cost + power_usage + missed_high_cost
        missed_high_cost = missed_high_cost / tot
        inference_cost = inference_cost / tot
        power_usage = power_usage / tot
        return power_usage, missed_high_cost, inference_cost

    def calc_reward(self, need_startup, missed_high, missed_medium, seen_highs, on_length, off_length):
        # number_of_camera = 4
        # assumed each inference takes 1/5 sec
        # startup_cost = inference_cost * 60
        # Cost of 1 watt of power
        # power_unit_cost = 0.0001452

        reward = -(self.norm_power_usage * on_length) - (missed_high * self.norm_missed_high_cost) - (
                on_length * self.norm_inference_cost)

        return reward

    def step(self, action):
        # Execute one time step within the environment
        # take action
        need_startup, missed_high, missed_medium, seen_highs, on_length, off_length = self._take_action(action)
        # calculate reward

        reward = self.calc_reward(need_startup, missed_high, missed_medium, seen_highs, on_length, off_length)
        # go to next step

        if self.mode == 2:
            ix = self.current_step
            output = "{},{},{},{}".format(self.df_list[self.df_index].loc[ix - 1, 'TimeStamp'], self.df_list[self.df_index].loc[ix, 'Label'],
                                       self.df_list[self.df_index].loc[ix, 'Percent'], action)
            self.test_logger.info(output)

        obs = self._next_observation()

        done = False
        if self.current_step == len(self.df_list[self.df_index]) - 1:
            done = True
        self.total_reward += reward
        return obs, reward, done, {"action": action, "label": self.df_list[self.df_index].loc[self.current_step, 'Label'],
                                   "day": self.df_list[self.df_index].loc[self.current_step, 'Day of week'],
                                   "hour": self.df_list[self.df_index].loc[self.current_step, 'Hour'],
                                   "min": self.df_list[self.df_index].loc[self.current_step, 'Minute']}

    def reset(self):
        # Start from the top if it's in testing or validation mode.
        if self.mode == 2 or self.mode == 1: 
            self.current_step = 0
            self.df_index = 0
        else:
            self.df_index = random.randint(0, len(self.df_list) - 1)
            self.current_step = random.randint(0, len(self.df_list[self.df_index]) - self.MAX_STEPS-1)

        

        if self.mode == 0:
            self.writer.add_scalar('Reward', self.total_reward, self.episode_num)
            self.episode_rewards.append(self.total_reward)
        self.episode_num += 1
        self.start_step = self.current_step
        self.total_off_time = 0
        self.total_on_time = 0
        self.total_time = 0
        self.total_highs = 0
        self.total_useless_on_time = 0
        self.total_startups = 0
        self.total_missed_highs = 0
        self.total_missed_mediums = 0
        self.total_seen_highs = 0
        self.total_reward = 0
        self.current_camera_state = True
        

        return self._next_observation()

    def render(self, mode='human', close=False):
        print(f'Env Version: {"1.7"}')
        print(f'Step: {self.current_step}')
        print(f'Total off time: {self.total_off_time}')
        print(f'Total on time: {self.total_on_time}')
        print(f'Total useless on time: {self.total_useless_on_time}')
        print(f'Total startups: {self.total_startups}')
        print(f'Total missed highs: {self.total_missed_highs}')
        print(f'Total missed mediums: {self.total_missed_mediums}')
        print(f'Total seen highs: {self.total_seen_highs}')
        print(f'Total number of high labels: {self.total_seen_highs + self.total_missed_highs}')

        if self.mode == 0:
            print(f'Total reward of the last episode: {self.episode_rewards[-1]}')
        print(f'Total reward of the current episode: {self.total_reward}')
        if self.total_highs:
            print(f'seen percentage: {self.total_seen_highs/self.total_highs}')

        print("------------------------------------\n")

    def get_run_stats(self):
        dic = {"steps": self.current_step,
            "off_time": self.total_off_time,
            "on_time": self.total_on_time,
            "useless_on": self.total_useless_on_time,
            "startups": self.total_startups,
            "missed_high": self.total_missed_highs,
            "total_seen_highs": self.total_seen_highs,
            "total_highs": self.total_seen_highs + self.total_missed_highs,
            "accuracy" : self.total_seen_highs/self.total_highs, 
            }
        return dic

class SolarStreetParkingEnv(gym.Env):

    def __init__(self):
        super(SolarStreetParkingEnv, self).__init__()
        self.total_reward = 0
        self.current_step = 0
        self.under_threshold_cnt = 0
        self.current_camera_state = True
        self.data_resolution_minutes = 1
        self.history_len = 60 + 1   #history used in rl is 6 box of average 10 minutes(60 minutes in total)
        self.cloud_coverage_len = 7
        self.history = []
        self.max_low_battery_penalty = 10.
        self.low_battery_threshold = 0.05
        self.init_history()

        self.max_battery_cap = 50 # at 5v
        self.camera_power_usage = 2.5 # max power of usb 1
        self.transmission_cost = 2.5 # max power of usb 1
        self.active_state_base_power_usage = 1
        self.deep_sleep_base_power_usage = 0.2 # this is the lowest power consumption of our system, when jetson is on but not doing anything, based on their doc it should in [0.5, 1.25]
        self.rl_inference_cost = 0.5 #cost of running rl on top of base_power_usage
        self.vision_inference_cost = 2.  #cost of running vision on top of base_power_usage
        self.missed_high_cost = 1.5
        self.solar_rating = 6  # amount of power the solar can generate in the perfect condition
        self.starting_battery_level = 1.

        self.camera_eusage_per_on = (self.camera_power_usage/(60 * 10)) # camera works for 6 seconds
        self.vision_eusage_per_on = (self.vision_inference_cost/(60 * 60)) # vision infrence takes 1 seconds
        self.transmission_eusage_per_on = self.transmission_cost/(60 * 60 * 20) # takes 50ms
        self.base_active_eusage_per_on = (self.active_state_base_power_usage)/ 60 

        self.base_sleep_eusage_per_off = (self.deep_sleep_base_power_usage/60) - (self.deep_sleep_base_power_usage/(60 * 6))
        self.rl_eusage_per_step = (self.rl_inference_cost/(60 * 500)) 
         
        self.inference_cost = self.camera_eusage_per_on + self.vision_eusage_per_on + \
                            self.transmission_eusage_per_on + self.base_active_eusage_per_on
                              
        self.power_usage = self.base_sleep_eusage_per_off + self.rl_eusage_per_step 

        self.norm_power_usage, self.norm_missed_high_cost, self.norm_inference_cost, self.normalized_max_low_battery_penalty = \
                            self.normalize(self.power_usage, self.missed_high_cost, self.inference_cost, self.max_low_battery_penalty)  
        
        # Action space consist of one value:
        # 0 means off and 1 means on
        self.action_space = gym.spaces.Discrete(2)

        # Observation contains:
        # day_of_week[0], day_of_week[1]: Cyclic represnetation of day of the week. Each value is in [-1., 1.]
        # time[0], time[1]: Cyclic represnetation of time of the day. Each value is in [-1., 1.]
        # history[i][0], history[i][1] for i in [0, 5]: history[i][0] is 0 or 1 showing whether the percentage is known or not. history[i][1] is the occupancy percentage [0., 1.] and it's 0 for unknown values.
        # battery remaining (watts): between [0 max_battery_cap]
        # daily charging power adjusted with weather: each has a value in range [0, self.solar_rating]
        # is_known: Whether curent percentage is known or not. It's 0 or 1
        # occupancy percent: Occupancy percent. In [0., 1.]
        # camera_state[0], camera_state[1]: shows if camera is on or off after the end of current action. If on camera_state[0] is 1 and camera_state[1] is 0, other wise they are 0 and 1 respectively. 

        lows = np.array([-1, -1, #day_of_week
                         -1, -1, #time
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, #history
                         0, #is_known
                         0, #battery remaining (watt)
                         0, 0, 0, 0, 0, 0, 0, # daily charging power adjusted with weather
                         0, #occupancy percent
                         0, 0#camera_state
                        ])
        highs = np.array([1, 1, #day_of_week
                         1, 1, #time
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, #history
                         1, #is_known
                         self.max_battery_cap, #battery remaining (watt)
                         self.solar_rating, self.solar_rating, self.solar_rating, self.solar_rating, self.solar_rating, self.solar_rating, self.solar_rating,# daily charging power adjusted with weather
                         1, #occupancy percent
                         1, 0#camera_state
                        ])
        self.observation_space = gym.spaces.Box(low=lows, high=highs, dtype=np.float32)

    def my_init(self, df_list, daily_weather, hourly_weather, debug=False, interval_len=1, mode=0, training_max_steps= 7 * 24 * 60, streets=[], is_tensorflow = False):
        self.daily_weather = daily_weather
        self.hourly_weather = hourly_weather
        self.MAX_STEPS = training_max_steps
        self.df_list = df_list
        self.debug = debug
        self.interval_len = interval_len
        self.mode = mode
        self.df_index = 0
        self.streets = streets
        self.is_tensorflow = is_tensorflow
        self.episode_num = 0
        self.steps_in_recovery = 0
        self.in_battery_recovery = False
        if mode == 0:
            self.writer = SummaryWriter()
        # create file handler which logs debug messages

        if mode == 2:
            path = 'logs/parking_multi_street.csv'
            logger_name = 'multifile'
            if len(streets) == 1:
                logger_name = str(streets[0])
                path = 'logs/parking_{}.csv'.format(streets[0])

            self.test_logger = logging.getLogger(logger_name)
            self.test_logger.setLevel(logging.INFO)
            fh = logging.FileHandler(path)
            fh.setLevel(logging.INFO)
            # add the handlers to the logger
            self.test_logger.addHandler(fh)

        for i in range(len(self.df_list)): 
            self.df_list[i] = self.df_list[i].reset_index(drop=True)

    def init_history(self):
        for i in range(0, self.history_len):
            self.history.append([0, 0])

    def add_to_history(self, is_known, precentage):
        self.history.pop(0)
        self.history.append([is_known, precentage])

        if len(self.history) != self.history_len:
            print("Wrong history length")

    def get_solar_efficiency(self):
        idx = int(self.current_step / 60)
        if idx >= len(self.hourly_weather):
            print("out of bound: {}/{}".format(idx, len(self.hourly_weather)))
            idx -= 1
        efficiency = self.hourly_weather.iloc[idx]['solarGenPrc']
        return efficiency

    def get_energy_production_prediction(self):
        idx = int(self.current_step / (60 * 24))
        if idx + self.cloud_coverage_len <= len(self.daily_weather):
            self.energy_production_pred = [self.daily_weather.iloc[idx + i]['solarGenPrc'] for i in range(self.cloud_coverage_len)]
        else:
            self.energy_production_pred.pop(0)
            self.energy_production_pred.append(0)
        return self.energy_production_pred

    def simulate_battery_usage(self, action):
        unfinished = False
        # on
        if action == 2:
            #/60 because action is for 1 minutes
            self.battery_rem -= self.camera_eusage_per_on  # camera, *10 is because we run it for 6 seconds in each minute
            self.battery_rem -= self.vision_eusage_per_on
            self.battery_rem -= self.transmission_eusage_per_on # transimision  cost for sending parking data, *60*20 because it takes 50 mili second to send inference
            self.battery_rem -= self.base_active_eusage_per_on
        elif action==1: 
            self.battery_rem -= self.base_sleep_eusage_per_off
            
        elif action == 0:
            self.steps_in_recovery += 1
            # recovery, device is off.
        else:
            print("wrong action")
            exit()
        if action > 0:
            self.battery_rem -= self.rl_eusage_per_step  
        
        if self.battery_rem < 0:
            unfinished = True
            self.battery_rem = 0

        # charging
        eff = self.get_solar_efficiency()
        charging_amount = (eff * self.solar_rating) / 60 # for a minute
        self.battery_rem += charging_amount
        self.battery_rem = min(self.battery_rem, self.max_battery_cap)

        if self.get_battery_prc() < self.low_battery_threshold:
            self.under_threshold_cnt += 1

        return unfinished

    def get_battery_rem(self):
        return self.battery_rem

    def get_battery_prc(self):
        battery_rem = self.get_battery_rem()
        
        return battery_rem/self.max_battery_cap

    def get_history(self):
        hist = []
        sum = 0
        cnt = 0

        for i in range(0, self.history_len - 1):
            # Precentage is known
            if self.history[i][0] == 1:
                cnt += 1
                sum += self.history[i][1]

            if (i + 1) % 10 == 0:
                if cnt > 0:
                    hist.append([1, sum / cnt])
                else:
                    hist.append([0, 0])

                sum = 0
                cnt = 0

        return hist

    def sin_cos(self, v):
        t = 2. * math.pi * v
        return (math.sin(t), math.cos(t))

    def _next_observation(self):
        total_passed_minutes = int(self.df_list[self.df_index].iloc[self.current_step]['Hour']) * 60 + int(
            self.df_list[self.df_index].iloc[self.current_step]['Minute'])
        camera_state = [1, 0]

        if self.current_camera_state:
            camera_state = [0, 1]

        is_known = 0
        occupancy_precent = 0

        if self.current_camera_state == True:
            is_known = 1
            occupancy_precent = self.df_list[self.df_index].iloc[self.current_step]['Percent']

        day_of_week = self.sin_cos(int(self.df_list[self.df_index].iloc[self.current_step]['Day of week']) / 6)
        time = self.sin_cos(total_passed_minutes / 1440)

        self.add_to_history(is_known, occupancy_precent)
        hist = self.get_history()

        battery_rem = self.get_battery_rem()

        obs = (day_of_week[0], day_of_week[1],
         time[0], time[1],
         hist[0][0], hist[0][1],
         hist[1][0], hist[1][1],
         hist[2][0], hist[2][1],
         hist[3][0], hist[3][1],
         hist[4][0], hist[4][1],
         hist[5][0], hist[5][1],
         is_known, 
         battery_rem,
         self.energy_production_pred[0], self.energy_production_pred[1], self.energy_production_pred[2],
         self.energy_production_pred[3], self.energy_production_pred[4], self.energy_production_pred[5],
         self.energy_production_pred[6],
         occupancy_precent,
         camera_state[0], camera_state[1])

        if self.is_tensorflow == False:
            obs = [torch.tensor(x, dtype=torch.float) for x in obs]
            obs = torch.stack(obs, dim=0)

        return obs

    def _take_action(self, action):
        action += 1

        missed_high = 0
        seen_high = 0
        missed_medium = 0
        on_length = 0
        off_length = 0
        need_startup = 0

        is_unfinished = self.simulate_battery_usage(action)

        if is_unfinished:
            action = 1

        if self.in_battery_recovery == True and self.get_battery_prc() >= self.low_battery_threshold:
            self.in_battery_recovery = False

        if action > 2:
            print("WRONG Action !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            exit()

        if action == 2:
            if self.current_camera_state == False:
                need_startup = 1

            self.current_camera_state = True
            on_length = self.interval_len

            index = self.current_step + 1
            label = self.df_list[self.df_index].iloc[index]['Label']
            self.total_highs += (label == 2)

            if label == 2:
                seen_high += 1

        else:
            self.current_camera_state = False
            off_length = self.interval_len

            index = self.current_step + 1
            label = self.df_list[self.df_index].iloc[index]['Label']
            self.total_highs += (label == 2)

            if label == 2:
                missed_high += 1
            elif label == 1:
                missed_medium += 1

        self.get_energy_production_prediction()

        if on_length > 0 and seen_high == 0:
            self.total_useless_on_time += on_length

        self.total_time += self.interval_len
        self.current_step += 1
        self.total_off_time += off_length
        self.total_on_time += on_length
        self.total_startups += need_startup
        self.total_missed_highs += missed_high
        self.total_missed_mediums += missed_medium
        self.total_seen_highs += seen_high

        return need_startup, missed_high, missed_medium, seen_high, on_length, off_length, is_unfinished

    def normalize(self, power_usage, missed_high_cost, inference_cost, max_low_battery_penalty):
        tot = inference_cost + power_usage + missed_high_cost + max_low_battery_penalty
        missed_high_cost = missed_high_cost / tot
        inference_cost = inference_cost / tot
        power_usage = power_usage / tot
        max_low_battery_penalty = max_low_battery_penalty/tot
        return power_usage, missed_high_cost, inference_cost, max_low_battery_penalty

    def calc_reward(self, need_startup, missed_high, missed_medium, seen_highs, on_length, off_length, is_unfinished):
        low_battery_reward = 0

        battery_prc = self.get_battery_prc()

        if battery_prc < self.low_battery_threshold:

            x = (self.low_battery_threshold - battery_prc)/self.low_battery_threshold
            coef = (pow(5, x) - 1)/(5-1)

            low_battery_reward = coef * self.normalized_max_low_battery_penalty

        missed_event_reward = missed_high * self.norm_missed_high_cost
        inference_reward = on_length * self.norm_inference_cost
        operation_base_reward = on_length * self.norm_power_usage
    
        reward = -operation_base_reward - missed_event_reward - inference_reward - low_battery_reward

        return reward

    def step(self, action):
        # Execute one time step within the environment
        if self.in_battery_recovery == True:
            action = -1
        
        # take action
        need_startup, missed_high, missed_medium, seen_highs, on_length, off_length, is_unfinished = self._take_action(action)

        if is_unfinished:
            self.in_battery_recovery = True
            action = 1

        # calculate reward
        reward = self.calc_reward(need_startup, missed_high, missed_medium, seen_highs, on_length, off_length, is_unfinished)
        # go to next step

        if self.mode == 2:
            ix = self.current_step
            solar_production = self.solar_rating * (self.get_solar_efficiency()/60)
            output = "{},{},{},{},{},{}".format(self.df_list[self.df_index].iloc[ix - 1]['TimeStamp'], self.df_list[self.df_index].iloc[ix]['Label'],
                                       self.df_list[self.df_index].iloc[ix]['Percent'], action, self.battery_rem, solar_production)
            self.test_logger.info(output)

        obs = self._next_observation()

        done = False
        if self.current_step == len(self.df_list[self.df_index]) - 1:
            done = True
        elif is_unfinished and self.mode != 2:
            done = True

        self.total_reward += reward
        return obs, reward, done, {"action": action, "label": self.df_list[self.df_index].iloc[self.current_step]['Label'],
                                   "day": self.df_list[self.df_index].iloc[self.current_step]['Day of week'],
                                   "hour": self.df_list[self.df_index].iloc[self.current_step]['Hour'],
                                   "min": self.df_list[self.df_index].iloc[self.current_step]['Minute'],
                                   "is_unfinished": is_unfinished}

    def reset(self):
        if self.mode == 0:
            self.writer.add_scalar('Reward', self.total_reward, self.episode_num)
            self.writer.add_scalar('steps_in_episode', self.current_step, self.episode_num)

        # Start from the top if it's in testing or validation mode.
        if self.mode == 2 or self.mode == 1: 
            self.current_step = 0
            self.df_index = 0
        else:
            self.df_index = random.randint(0, len(self.df_list) - 1)
            self.current_step = random.randint(0, len(self.df_list[self.df_index]) - self.MAX_STEPS-1)

        self.episode_num += 1
        self.start_step = self.current_step
        self.total_off_time = 0
        self.total_on_time = 0
        self.total_time = 0
        self.total_highs = 0
        self.total_useless_on_time = 0
        self.total_startups = 0
        self.total_missed_highs = 0
        self.total_missed_mediums = 0
        self.total_seen_highs = 0
        self.total_reward = 0
        self.steps_in_recovery = 0
        self.current_camera_state = True
        self.in_battery_recovery = False
        self.battery_rem = self.starting_battery_level * self.max_battery_cap
        self.get_energy_production_prediction()

        return self._next_observation()

    def render(self, mode='human', close=False):
        print(f'Env Version: {"1.7"}')
        print(f'Step: {self.current_step}')
        print(f'Battery: {self.battery_rem}')
        print(f'Total off time: {self.total_off_time}')
        print(f'Total on time: {self.total_on_time}')
        print(f'Total useless on time: {self.total_useless_on_time}')
        print(f'Total startups: {self.total_startups}')
        print(f'Total missed highs: {self.total_missed_highs}')
        print(f'Total missed mediums: {self.total_missed_mediums}')
        print(f'Total seen highs: {self.total_seen_highs}')
        print(f'Total number of high labels: {self.total_seen_highs + self.total_missed_highs}')

        print(f'Total reward of the current episode: {self.total_reward}')
        if self.total_highs:
            print(f'seen percentage: {self.total_seen_highs/self.total_highs}')
        print("Number of times battery was under threshold {}".format(self.under_threshold_cnt))
        print("Steps in battery recovery: ", self.steps_in_recovery)

        print("------------------------------------\n")

    def print_stats(self):
        total_highs = 0
        for index, row in self.df_list[self.df_index].iterrows():
            total_highs+= row['Label']==2

        print("total highs = ", total_highs)

    def get_all_high_rows(self, idx):
        high_idx = self.df_list[idx]['Label'] == 2
        return self.df_list[idx][high_idx]

    def get_high_counts_in_data(self, idx):
        high_idx = self.df_list[idx]['Label'] == 2
        return len(self.df_list[idx][high_idx])

    def get_current_row(self):
        return self.df_list[self.df_index].iloc[self.current_step]

    def get_run_stats(self):
        dic = {"steps": self.current_step,
            "off_time": self.total_off_time,
            "on_time": self.total_on_time,
            "useless_on": self.total_useless_on_time,
            "startups": self.total_startups,
            "missed_high": self.total_missed_highs,
            "total_seen_highs": self.total_seen_highs,
            "total_highs": self.total_seen_highs + self.total_missed_highs,
            "accuracy" : self.total_seen_highs/self.total_highs, 
            "steps_under_threshold" : self.under_threshold_cnt,
                "steps_in_recovry" : self.steps_in_recovery
            }
        return dic