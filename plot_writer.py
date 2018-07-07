import matplotlib.pyplot as plt


class PlotWriter(object):
    """docstring for PlotWriter"""

    def __init__(self, config):
        super(PlotWriter, self).__init__()
        self.config = config

    def write(self, predictor):
        plt.plot(list(range(len(predictor.data_all))), predictor.data_all, 'b', label='true')
        plt.plot(list(range(predictor.n_train)), predictor.pred_train, 'g', label='pred_train')
        plt.plot(list(range(predictor.n_train)), predictor.pred_train_lower, 'darkgreen', label='pred_train_lower')
        plt.plot(list(range(predictor.n_train)), predictor.pred_train_upper, 'lime', label='pred_train_upper')
        plt.plot(list(range(predictor.n_train, len(predictor.data))), predictor.pred_valid, 'r', label='pred_valid')
        plt.plot(list(range(predictor.n_train, len(predictor.data))), predictor.pred_valid_lower, 'darkred',
                 label='pred_valid_lower')
        plt.plot(list(range(predictor.n_train, len(predictor.data))), predictor.pred_valid_upper, 'salmon',
                 label='pred_valid_upper')
        plt.plot(list(range(len(predictor.data), len(predictor.data) + len(predictor.pred_test))), predictor.pred_test, 'c',
                 label='pred_test')
        plt.plot(list(range(len(predictor.data), len(predictor.data) + len(predictor.pred_test))), predictor.pred_test_lower,
                 'darkcyan',
                 label='pred_test_lower')
        plt.plot(list(range(len(predictor.data), len(predictor.data) + len(predictor.pred_test))), predictor.pred_test_upper,
                 'aquamarine',
                 label='pred_test_upper')

        plt.plot([], [], ' ', label='mse_train: {0:.2f}'.format(predictor.mse_train))
        plt.plot([], [], ' ', label='mse_valid: {0:.2f}'.format(predictor.mse_valid))
        plt.plot([], [], ' ', label='mse: {}'.format(predictor.mse))
        plt.legend()

        plt.title('Forecast for ' + predictor.basename)
        plt.savefig(self.config.fig_prefix + predictor.basename.replace('json', 'pdf'))
        plt.close()
