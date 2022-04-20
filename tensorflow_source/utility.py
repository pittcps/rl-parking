
import os
import gin
import json
import pandas as pd
from environment import parking_env

@gin.configurable
def set_hyper_param(mem_size, train_end, validation_end, episodes, episode_max_steps, batch_size, lr, \
                    street_list, is_tensorflow, gpu_id, parking_data_dir, weather_data_dir, mode):
    batches_per_epoch = episode_max_steps/batch_size
    total_steps = episodes * train_end

    eps_last_frame = train_end*episodes
    
    return {'mem_size': mem_size, 
            'train_end': train_end - 60,
            'validation_end': validation_end - 60,
            'episodes': episodes, 
            'episode_max_steps': episode_max_steps,
            'batch_size': batch_size,
            'lr': lr,
            'street_list': street_list,
            'batches_per_epoch': batches_per_epoch, 
            'total_steps': total_steps,
            'eps_last_frame': eps_last_frame,
            'is_tensorflow': is_tensorflow,
            'gpu_id': gpu_id,
            'parking_data_dir' : parking_data_dir,
            'weather_data_dir' : weather_data_dir,
            'mode': mode
            }

def create_environments(street_list, train_end, validation_end, episode_max_steps, parking_data_dir, is_tensorflow):
    train_env = parking_env.VanillaStreetParkingEnv()
    validation_envs = []
    test_envs = []

    df_train_list = []

    for i in range(len(street_list)):
        df = pd.read_csv(os.path.join(parking_data_dir, "street/one_street_data_{}.csv".format(street_list[i])))
        df_train_list.append(df[:train_end])
        
        validation_env = parking_env.VanillaStreetParkingEnv()
        validation_env.my_init(df_list=[df[train_end:validation_end]], mode = 1, streets=[street_list[i]], is_tensorflow=is_tensorflow)
        
        test_env = parking_env.VanillaStreetParkingEnv()
        test_env.my_init(df_list=[df[validation_end:]], mode=2, streets=[street_list[i]], is_tensorflow=is_tensorflow)

        test_envs.append(test_env)
        validation_envs.append(validation_env)

    train_env.my_init(df_list=df_train_list, streets=street_list, training_max_steps=episode_max_steps, is_tensorflow=is_tensorflow)

    return train_env, validation_envs, test_envs

def create_season_aware_environments(street_list, episode_max_steps, parking_data_dir, weather_data_dir, is_tensorflow):
    f = open(os.path.join(parking_data_dir, 'data_splits.json'))
    split_idx = json.load(f)
    f.close()

    train_hourly_idx =  [split_idx['train'][x]/(60) for x in range(0, len(split_idx['train']), 60)]
    val_hourly_idx = [split_idx['val'][x]/(60) for x in range(0, len(split_idx['val']), 60)]
    test_hourly_idx = [split_idx['test'][x]/(60) for x in range(0, len(split_idx['test']), 60)]


    train_daily_idx = [split_idx['train'][x]/(60 * 24) for x in range(0, len(split_idx['train']), 60 * 24)]
    val_daily_idx = [split_idx['val'][x]/(60 * 24) for x in range(0, len(split_idx['val']), 60 * 24)]
    test_daily_idx = [split_idx['test'][x]/(60 * 24) for x in range(0, len(split_idx['test']), 60 * 24)]

    hourly_solar_weather_data_dir = os.path.join(weather_data_dir, "weather_data/solar_weather_data_new_date.csv") 
    daily_solar_weather_data_dir = os.path.join(weather_data_dir, "weather_data/solar_weather_daily_data_new_date.csv")

    df_sw_hourly = pd.read_csv(hourly_solar_weather_data_dir)
    df_sw_daily = pd.read_csv(daily_solar_weather_data_dir)

    df_sw_hourly = pd.read_csv(hourly_solar_weather_data_dir)

    weather_h_train = df_sw_hourly.iloc[train_hourly_idx]
    weather_h_train.reset_index(drop=True, inplace=True)
    weather_h_val = df_sw_hourly.iloc[val_hourly_idx]
    weather_h_val.reset_index(drop=True, inplace=True)
    weather_h_test = df_sw_hourly.iloc[test_hourly_idx]
    weather_h_test.reset_index(drop=True, inplace=True)

    df_sw_daily = pd.read_csv(daily_solar_weather_data_dir)
    weather_d_train = df_sw_daily.iloc[train_daily_idx]
    weather_d_train.reset_index(drop=True, inplace=True)
    weather_d_val = df_sw_daily.iloc[val_daily_idx]
    weather_d_val.reset_index(drop=True, inplace=True)
    weather_d_test = df_sw_daily.iloc[test_daily_idx]
    weather_d_test.reset_index(drop=True, inplace=True)
    
    train_env = parking_env.SolarStreetParkingEnv()
    validation_envs = []
    test_envs = []

    df_train_list = []

    for i in range(len(street_list)):
        df = pd.read_csv(os.path.join(parking_data_dir, "street/one_street_data_{}.csv".format(street_list[i])))
        train_df = df.iloc[split_idx['train']]
        train_df.reset_index(drop=True, inplace=True)
        df_train_list.append(train_df)

        validation_env = parking_env.SolarStreetParkingEnv()
        val_df = df.iloc[split_idx['val']]
        val_df.reset_index(drop=True, inplace=True)
        validation_env.my_init(df_list=[val_df], daily_weather=weather_d_val, hourly_weather=weather_h_val, mode = 1, streets=[street_list[i]], is_tensorflow=is_tensorflow)

        test_env = parking_env.SolarStreetParkingEnv()
        test_df = df.iloc[split_idx['test']]
        test_df.reset_index(drop=True, inplace=True)
        test_env.my_init(df_list=[test_df], daily_weather=weather_d_test, hourly_weather=weather_h_test, mode=2, streets=[street_list[i]], is_tensorflow=is_tensorflow)

        test_envs.append(test_env)
        validation_envs.append(validation_env)

    

    train_env.my_init(df_list=df_train_list, daily_weather=weather_d_train, hourly_weather=weather_h_train, streets=street_list, training_max_steps=episode_max_steps, is_tensorflow=is_tensorflow)

    return train_env, validation_envs, test_envs


def create_syntethic_test_env(street_list, n_sunny, n_overcast, sunny_first, parking_data_dir, weather_data_dir, is_tensorflow):
    if sunny_first:
        hourly_solar_weather_data_dir = os.path.join(weather_data_dir, 'weather_data/synthetic/{}sunny_{}overcast_solar_weather_data.csv'.format(n_sunny, n_overcast))
        daily_solar_weather_data_dir = os.path.join(weather_data_dir, 'weather_data/synthetic/daily_{}sunny_{}overcast_solar_weather_data.csv'.format(n_sunny, n_overcast))
    else:
        hourly_solar_weather_data_dir = os.path.join(weather_data_dir, 'weather_data/synthetic/{}overcast_{}sunny_solar_weather_data.csv'.format(n_overcast, n_sunny))
        daily_solar_weather_data_dir = os.path.join(weather_data_dir, 'weather_data/synthetic/daily_{}overcast_{}sunny_solar_weather_data.csv'.format(n_overcast, n_sunny))

    hourly_solar_weather_df = pd.read_csv(hourly_solar_weather_data_dir)
    daily_solar_weather_df = pd.read_csv(daily_solar_weather_data_dir)

    start = 3*24*7*60

    test_envs = []

    for i in range(len(street_list)):
        df = pd.read_csv(os.path.join(parking_data_dir, "street/one_street_data_{}.csv".format(street_list[i])))
        df = df[start:start+(n_sunny + n_overcast) * 60 * 24]
        df.reset_index(drop=True, inplace=True)

        test_env = parking_env.SolarStreetParkingEnv()
        test_env.my_init(df_list=[df], daily_weather=daily_solar_weather_df, hourly_weather=hourly_solar_weather_df, mode=2, streets=[street_list[i]], is_tensorflow=is_tensorflow)

        test_envs.append(test_env)

    return test_envs

def create_week_test_env(street_list, use_best, parking_data_dir, weather_data_dir, is_tensorflow):
    if use_best:
        name = "best"
    else:
        name = "worst"
        
    hourly_solar_weather_data_dir = os.path.join(weather_data_dir, 'weather_data/synthetic/{}_week_hourly.csv'.format(name))
    daily_solar_weather_data_dir = os.path.join(weather_data_dir, 'weather_data/synthetic/{}_week_daily.csv'.format(name))

    hourly_solar_weather_df = pd.read_csv(hourly_solar_weather_data_dir)
    daily_solar_weather_df = pd.read_csv(daily_solar_weather_data_dir)

    start = 3*24*7*60

    test_envs = []

    for i in range(len(street_list)):
        df = pd.read_csv(os.path.join(parking_data_dir, "street/one_street_data_{}.csv".format(street_list[i])))
        df = df[start:start+ len(daily_solar_weather_df) * 60 * 24]
        df.reset_index(drop=True, inplace=True)

        test_env = parking_env.SolarStreetParkingEnv()
        test_env.my_init(df_list=[df], daily_weather=daily_solar_weather_df, hourly_weather=hourly_solar_weather_df, mode=2, streets=[street_list[i]], is_tensorflow=is_tensorflow)

        test_envs.append(test_env)

    return test_envs

def create_unseen_test_env(street_list, parking_data_dir, weather_data_dir, is_tensorflow):
    hourly_solar_weather_data_dir = os.path.join(weather_data_dir, "weather_data/solar_weather_data_new_date.csv") 
    daily_solar_weather_data_dir = os.path.join(weather_data_dir, "weather_data/solar_weather_daily_data_new_date.csv")

    hourly_solar_weather_df = pd.read_csv(hourly_solar_weather_data_dir)
    daily_solar_weather_df = pd.read_csv(daily_solar_weather_data_dir)


    test_envs = []

    for i in range(len(street_list)):
        df = pd.read_csv(os.path.join(parking_data_dir, "street/one_street_data_{}.csv".format(street_list[i])))

        test_env = parking_env.SolarStreetParkingEnv()
        test_env.my_init(df_list=[df], daily_weather=daily_solar_weather_df, hourly_weather=hourly_solar_weather_df, mode=2, streets=[street_list[i]], is_tensorflow=is_tensorflow)

        test_envs.append(test_env)

    return test_envs
