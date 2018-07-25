class Config(object):
    def __init__(self):
        super(Config, self).__init__()

        self.upload_prefix = 'upload/'
        self.data_prefix = 'data/'
        self.model_prefix = 'model/'
        self.fig_prefix = 'fig/'
        self.output_prefix = 'output/'
        self.log_prefix = 'log/'

        self.n_batch = 1
        self.n_max_epoch = 1000
        self.n_neurons = 256
        self.n_predict_step = 10
        self.n_input_dim = 1
        self.n_output_dim = 30
        self.n_patience = 50
        self.n_lr_decay = 2
        self.lr = 1e-3
        self.lr_decay = 0.9
        self.ratio_valid = 0.1
        self.max_valid = 100
        self.valid_loss_weight = 0.6
        self.test_split = True
        self.mse_threshold = 0.2
