import re
from datetime import datetime
from dateutil.relativedelta import relativedelta


arbitrary_interval = re.compile(r"(?P<howmany>\d+)(?P<type>\w)")

def sanitize_values(sepspec):
    """
    This is to overcome some of the issues encountered in a separation spec, like commas as number separators.
    :param sepspec:
    The separation spec.
    :return:
    A sanitized version of the spec
    """
    for i in range(len(sepspec['values'])):
        sepspec['values'][i] = sepspec['values'][i].replace(",", ".")
    return sepspec

def string2interval(s):
    """
    This converts a string to a relativedelta.
    :param s:
    This accepts this:
        raw deltas:
            month
            day
        multiples:
            Xm
            Xd
            ... where X is a positive integer
    :return:
    the delta
    """
    m = arbitrary_interval.match(s)

    if m is None:
        x = 1
        if s == "week":
            x = 7
            s = "day"
    else:
        m = m.groupdict()
        x = int(m['howmany'])
        s = "month" if m['type'] == 'm' else 'day'

    if s == "month":
        return relativedelta(months=x)
    elif s == "day":
        return relativedelta(days=x)


def aditya_interpolator(timeseries, interval, invert_time_series=False):
    """
    This is a script to go through a timeseries object and to make it into a final timeseries object.
    If an entry is null, it will be overwritten by the last value.
    The function also makes sure that all
    :param timeseries:
    the timeseries object. formatted after how pedro sent me the dates through Slack.
        {
          "dimensions": [
            {
              "type": "int",
              "semantic_type": "count"
            },
            {
              "type": "float",
              "semantic_type": "price_usd"
            }
          ],
          "ts": [
            [
              "2017-08-01T00:00:00.000Z",
              1,
              1311.75
            ],
            [
              "2017-09-01T00:00:00.000Z",
              21,
              27614.550000000003
            ],
            [
              "2017-10-01T00:00:00.000Z",
              22,
              28149.300000000003
            ],
            [
              "2017-11-01T00:00:00.000Z",
              22,
              28210.25
            ],
            [
              "2017-12-01T00:00:00.000Z",
              19,
              21441.350000000002
            ],
            [
              "2018-01-01T00:00:00.000Z",
              14,
              18535.05
            ]
          ],
          "metadata": {}
        }
    :param ifp:
        The IFP object. needed to make the dimensions object
    :param interval:
        The interval of the time readings.
        This should be a
    :param invert_time_series:
        Whether or not to invert the time series 1/x for x in series.
    :return:
        A reformatted time series object.
    """
    ijump = string2interval(interval)

    # make a dictionary of the dataset. this is useful for identifying the right record.
    # date -> record
    time_format = "%Y-%m-%dT%H:%M:%S.000Z"
    dsts = {}
    for ts in timeseries['ts']:
        dt = datetime.strptime(ts[0], time_format)
        if invert_time_series and ts[-1] != 0 and ts[-1] is not None:
            ts[-1] = 1. / ts[-1]
        dsts[dt] = [ts[0], ts[-1]]

    final_ts = []
    last_val = None

    val = min(dsts.keys())
    last_time = max(dsts.keys())
    # inspect each item at the interval marked
    while val <= last_time:
        record = dsts.get(val, [val.strftime(time_format), last_val])
        if record[-1] is None:
            record[-1] = last_val
        final_ts.append(record)
        last_val = final_ts[-1][-1]
        val += ijump

    return final_ts


def toandy(orig, ifp, interval, lasteventdate=None, invert_time_series=False, fake_a_date=False):
    ifptype = "Boolean" if "::Binary" in ifp['type'] else "Ordered Multinomial"

    if 'parsed_answers' not in ifp['ifp']:
        return {"ifp": ifp, "payload": "Invalid IFP"}, "Invalid"

    import json
    open("262.json", "w").write(json.dumps(ifp))
    root = {
        "ifp": ifp,
        "payload": {
            "question_type": ifptype,
            "separations": sanitize_values(ifp['ifp']['parsed_answers']),
            "historical_data": {
                "dimensions": ["DATE", orig["dimensions"][-1]],
                "ts": aditya_interpolator(orig, interval, invert_time_series)
            }
        }
    }

    # add a lst event date for event timeseries
    if lasteventdate is not None:
        root['payload']['last-event-date'] = lasteventdate

    if fake_a_date:
        onemonth = datetime.now() + relativedelta(months=1)
        root['ifp']['ends_at'] = onemonth.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    return root, "Valid"

