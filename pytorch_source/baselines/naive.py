
def train(env, idx):
    lh = 24
    uh = 0
    highs = env.get_all_high_rows(idx)

    lh = highs["Hour"].min()
    uh = highs["Hour"].max()

    print(lh, ' ', uh)
        
    return lh, uh

def test(lh, uh, env, end_with_battery = False, render = False, idx = 0):

    ds_idx = idx

    env.reset()
    env.current_step = 0
    env.df_index = ds_idx

    on_actions = 0

    while True:

        if env.current_step + 1 >= len(env.df_list[env.df_index]):
            break

        row = env.get_current_row()
        if row['Hour'] >= lh and row['Hour'] <= uh:
            action = 1
            on_actions += 1
        else:
            action = 0

        obs, reward, done, info = env.step(action)
        if render:
            env.render()
        if done and info['is_unfinished'] == False:
            break
        if done and end_with_battery:
            break


    print("{} steps out of {}".format(env.current_step, len(env.df_list[env.df_index])))
    env.render()

