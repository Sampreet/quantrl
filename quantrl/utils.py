def get_default_trajectories(env):
    env.action_interval = env.t_dim - 1
    for _ in range(env.n_trajectories):
        env.reset()
        env.step([1.0])
    if env.plot:
        env.plotter.show_plot()
    return env.io.all_data