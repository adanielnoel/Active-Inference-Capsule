class Args:
    def __init__(self,
                 settings='',
                 batch_agents=1,
                 max_cpu=-1,
                 make_video=False,
                 display_plots=False,
                 load_existing=False,
                 save_dirpath=None,
                 model_load_filepath=None,
                 save_all_episodes=False,
                 verbose=True,
                 display_simulation=False):
        self.settings = settings
        self.batch_agents = batch_agents
        self.max_cpu = max_cpu
        self.make_video = make_video
        self.display_plots = display_plots
        self.load_existing = load_existing
        self.save_dirpath = save_dirpath
        self.model_load_filepath = model_load_filepath
        self.save_all_episodes = save_all_episodes
        self.verbose = verbose
        self.display_simulation = display_simulation
