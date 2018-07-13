import json

import numpy as np
import scipy
import scipy.stats
from dateutil.relativedelta import relativedelta
import pdb

class JSONWriter(object):
    """docstring for JSONWriter"""

    def __init__(self, config):
        super(JSONWriter, self).__init__()
        self.config = config

    def write(self, predictor):
        pred = np.insert(predictor.pred_test, 0, predictor.data[-1])
        pred_lower = np.insert(predictor.pred_test_lower, 0, predictor.data[-1])
        pred_upper = np.insert(predictor.pred_test_upper, 0, predictor.data[-1])

        if self.config.test_split:
            last_date = predictor.dates[-(self.config.n_predict_step + 1)]
        else:
            last_date = predictor.dates[-1]

        diff = predictor.dates[-1] - predictor.dates[-2]
        if diff.days in (365, 366):
            diff = relativedelta(years=1)
        elif diff.days in (28, 29, 30, 31):
            diff = relativedelta(months=1)

        pred = pred.tolist()
        new_dates = [last_date + diff * i for i in range(len(pred))]
        new_dates_str = [d.strftime("%Y-%m-%d") for d in new_dates]

        if predictor.mse > self.config.mse_threshold:
            print('mse is too large')
            forecast_is_usable = 0
        else:
            forecast_is_usable = 1

        prob_list = self.calc_prob_option(predictor, predictor.pred_test[-1], predictor.pred_test_lower[-1], predictor.pred_test_upper[-1])

        res = {
            'forecast_is_usable': [forecast_is_usable],
            'forecasts': {
                'RNN': {
                    'forecast_is_usable': [forecast_is_usable],
                    'internal': {
                        'rmse': [predictor.mse]
                    },
                    'model': ['RNN'],
                    'to_date': [new_dates_str[-1]],
                    'ts': [[new_dates_str[i], str(pred[i]), str(pred_lower[i]), str(pred_upper[i])] for i in
                           range(len(pred))],
                    'ts_colnames': [
                        'date',
                        'Point Forecast',
                        'Lo 95',
                        'Hi 95'
                    ],
                    'option_labels': predictor.content['ifp']['ifp']['parsed_answers']['values'],
                    'option_probabilities': prob_list.tolist()
                }
            },
            'internal': {
                'rmse': [predictor.mse]
            },
            'model': ['RNN'],
            'option_labels': predictor.content['ifp']['ifp']['parsed_answers']['values'],
            'option_probabilities': prob_list.tolist(),
            'parsed_request': {
                'fcast_dates': new_dates_str[1:],
                'h': [self.config.n_predict_step],
                'target': {
                    'date': [d.strftime("%Y-%m-%d") for d in predictor.dates],
                    'value': predictor.data_all
                },
                'target_tail': {
                    'date': [predictor.dates[-1].strftime("%Y-%m-%d")],
                    'value': [predictor.data_all[-1]]
                }
            },
            'to_date': [new_dates_str[-1]],
            'ts': [[new_dates_str[i], str(pred[i]), str(pred_lower[i]), str(pred_upper[i])] for i in range(len(pred))],
            'ts_colnames': [
                'date',
                'Point Forecast',
                'Lo 95',
                'Hi 95'
            ],
        }

        outputname = predictor.basename.replace('_input_', '_output_')
        with open(self.config.output_prefix + outputname, 'w') as fout:
            json.dump(res, fout)

    def calc_prob_option(self, predictor, pred, pred_lower, pred_upper):
        if predictor.content['ifp']['ifp']['parsed_answers']['unit'] == 'boolean':
            value_list = [[None, 0.5], [0.5, None]]
        else:
            option_text = predictor.content['ifp']['ifp']['parsed_answers']['values']
            value_list = []

            for text in option_text:
                if text[0] == '<':
                    values = [None, float(text[1:])]
                elif text[0] == '>':
                    values = [float(text[1:]), None]
                else:
                    values = [float(t) for t in text.split(' - ')]

                value_list.append(values)

        rv_lower = scipy.stats.norm(loc=pred, scale=(pred - pred_lower) / 2)
        rv_upper = scipy.stats.norm(loc=pred, scale=(pred_upper - pred) / 2)

        def return_prob(value):
            if value <= pred:
                return rv_lower.cdf(value)
            else:
                return rv_upper.cdf(value)

        prob_list = []
        for v in value_list:
            low, high = v
            if low is None:
                prob_list.append(return_prob(high))
            elif high is None:
                prob_list.append(1 - return_prob(low))
            else:
                prob_list.append(return_prob(high) - return_prob(low))

        prob_list = np.asarray(prob_list)
        print(sum(prob_list))
        prob_list = prob_list / sum(prob_list)
        print(prob_list)
        return prob_list
