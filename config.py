class Config(object):
    def __init__(self):
        super(Config, self).__init__()

        self.model_prefix = 'model/'
        self.fig_prefix = 'fig/'
        self.output_prefix = 'output/'

        self.n_batch = 1
        self.n_max_epoch = 1000
        self.n_neurons = 32
        self.n_timestep = 1
        self.n_predict_step = 10
        self.n_input_dim = 1
        self.n_output_dim = 10
        self.n_patience = 10
        self.n_lr_decay = 2
        self.lr_decay = 0.99
        self.ratio_valid = 0.1
        self.max_valid = 10
        self.valid_loss_weight = 0.5
        self.test_split = True


