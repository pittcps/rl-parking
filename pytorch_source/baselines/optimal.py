

def simulate_perfect_knowledge_env(env, render = False, idx = 0):

    ds_idx = idx
    env.reset()
    env.current_step = 0
    env.df_index = ds_idx

    on_actions = 0

    while True:

        if env.current_step + 1 >= len(env.df_list[env.df_index]):
            break

        if env.df_list[env.df_index].loc[env.current_step + 1, "Label"] == 2:
            action = 1
            on_actions += 1
        else:
            action = 0

        obs, reward, done, info = env.step(action)
        if render:
            env.render()
        if done:
            break

    print("{} steps out of {}".format(env.current_step, len(env.df_list[env.df_index])))
    env.render()
