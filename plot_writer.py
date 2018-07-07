import matplotlib.pyplot as plt


class PlotWriter(object):
    """docstring for PlotWriter"""

    def __init__(self, config):
        super(PlotWriter, self).__init__()
        self.config = config

    def write(self, model):
        plt.plot(list(range(len(model.data_all))), model.data_all, 'b', label='true')
        plt.plot(list(range(model.n_train)), model.pred_train, 'g', label='pred_train')
        plt.plot(list(range(model.n_train)), model.pred_train_lower, 'darkgreen', label='pred_train_lower')
        plt.plot(list(range(model.n_train)), model.pred_train_upper, 'lime', label='pred_train_upper')
        plt.plot(list(range(model.n_train, len(model.data))), model.pred_valid, 'r', label='pred_valid')
        plt.plot(list(range(model.n_train, len(model.data))), model.pred_valid_lower, 'darkred',
                 label='pred_valid_lower')
        plt.plot(list(range(model.n_train, len(model.data))), model.pred_valid_upper, 'salmon',
                 label='pred_valid_upper')
        plt.plot(list(range(len(model.data), len(model.data) + len(model.pred_test))), model.pred_test, 'c',
                 label='pred_test')
        plt.plot(list(range(len(model.data), len(model.data) + len(model.pred_test))), model.pred_test_lower,
                 'darkcyan',
                 label='pred_test_lower')
        plt.plot(list(range(len(model.data), len(model.data) + len(model.pred_test))), model.pred_test_upper,
                 'aquamarine',
                 label='pred_test_upper')

        plt.plot([], [], ' ', label='mse_train: {0:.2f}'.format(model.mse_train))
        plt.plot([], [], ' ', label='mse_valid: {0:.2f}'.format(model.mse_valid))
        plt.plot([], [], ' ', label='mse: {}'.format(model.mse))
        plt.legend()

        plt.title('Forecast for ' + model.basename)
        plt.savefig(self.config.fig_prefix + model.basename.replace('json', 'pdf'))
        plt.close()
