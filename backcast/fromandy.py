import calendar
import time


def spans2cents(spans):
    # figure out the jump size
    mapped = []
    realign = []
    adiff = None
    mappref = {"Yes": 1, "No": 0}
    for s in spans:
        j = s.split(" - ")
        if len(j) > 1:
            start = float(j[0])
            end = float(j[1])
            mapped.append( (end + start) / 2. )
            realign.append(False)
            adiff = end - start
        else:
            if s in mappref:
                mapped.append(mappref[s])
                realign.append(False)
            else:
                smp = s.replace(">", "").replace("<", "")
                mapped.append(float(smp))
                realign.append(True)

    if adiff is not None and realign[0]:
        mapped[0] -= adiff/2.
    if adiff is not None and realign[-1]:
        mapped[0] += adiff/2.

    return mapped


def andyzipper(cvt, andy_response):
    if andy_response is None or 'r_error_message' in andy_response or andy_response['forecast_is_usable'][0] != 1:
        cvt['payload']['ts_pred'] = []
        cvt['payload']['ts_conf'] = []
        cvt['payload']['boxprobs'] = {}
        cvt['payload']['boxnames'] = []
        return cvt

    pred_ts = []
    pred_conf = []
    for t in andy_response['ts']:
        tm = calendar.timegm(time.strptime(t[0], '%Y-%m-%d'))
        tm = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(tm))
        pred_ts.append([tm, float(t[1])])
        pred_conf.append([tm, float(t[1]) - float(t[2])])

    cvt['payload']['ts_pred'] = pred_ts
    cvt['payload']['ts_conf'] = pred_conf

    cvt['payload']['boxnames'] = cvt['ifp']['ifp']['parsed_answers']['values']
    cvt['payload']['boxprobs'] = dict(zip(spans2cents(cvt['payload']['boxnames']),
                                             andy_response['option_probabilities']))

    return cvt
