import json
import re
import requests
import sys
from ifp_similarity_bank import IFPTemplateBank
from query_maker import IFPKGToQuery
from toandy import toandy
from fromandy import andyzipper
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pdb
import os

if len(sys.argv) < 3:
    # print("run_pipeline.py PROJECT_NAME MAX_SIMILARITY_SCORE")
    # sys.exit(1)
    PROJECT_NAME = "sage_kg_v2"
    MAX_SIMILARITY_SCORE = 8.0
else:
    PROJECT_NAME = sys.argv[1]
    MAX_SIMILARITY_SCORE = float(sys.argv[2])

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=os.cpu_count())


def process_timeseries(query, authdict, interval, offset):
    print("Processing a ts")
    request = requests.get(
        "http://sage-dev-internal.isi.edu/mydig/projects/%s/search/conjunctive?%s" % (PROJECT_NAME, query),
        headers=authdict)
    result = json.loads(request.text)

    rid = result['hits']['hits'][0]['_id']
    stuffurl = "http://sage-dev-internal.isi.edu/mydig/projects/%s/search/event?measure=" \
               "%s&_group-by=event_date&_aggregation=avg&_aggregation-field=value&_interval=%s" \
               "&event_date$greater-equal-than=1970-01-01&_offset=%s" % (
                   PROJECT_NAME, rid, interval, offset)
    print(stuffurl)

    request = requests.get(stuffurl, headers=authdict)
    result = json.loads(request.text)
    # print json.dumps(result, indent=2)
    return result


def process_timeseries_date(query, authdict, interval, offset,req_date):
    print("Processing a ts")
    request = requests.get(
        "http://sage-dev-internal.isi.edu/mydig/projects/%s/search/conjunctive?%s" % (PROJECT_NAME, query),
        headers=authdict)
    result = json.loads(request.text)

    rid = result['hits']['hits'][0]['_id']
    stuffurl = "http://sage-dev-internal.isi.edu/mydig/projects/%s/search/event?measure=" \
               "%s&_group-by=event_date&_aggregation=avg&_aggregation-field=value&_interval=%s" \
               "&event_date$greater-equal-than=1970-01-01&_offset=%s&event_date$less-equal-than=%s" % (
                   PROJECT_NAME, rid, interval, offset,req_date)
    # print(stuffurl)

    request = requests.get(stuffurl, headers=authdict)

    # if the granularity is not daily, then figure out the last *day* with an event
    if interval != 'day':
        stuffurl = stuffurl.replace("_interval=%s" % interval, "_interval=day")
        dayrequest = requests.get(stuffurl, headers=authdict)
    else:
        dayrequest = request

    # figure out the last event date
    result = json.loads(dayrequest.text)
    lasteventdate = max([i[0] for i in result['ts']])

    result = json.loads(request.text)

    return result, lasteventdate

def process_event(query, authdict, gran):
    print("Processing an event")
    stuffurl = "http://sage-dev-internal.isi.edu/mydig/projects/%s/search/event?%s" % (PROJECT_NAME, query)
    # print(stuffurl)

    request = requests.get(stuffurl, headers=authdict)
    if request.status_code == 404:
        return None, None

    # if the granularity is not daily, then figure out the last *day* with an event
    if gran != 'day':
        stuffurl = stuffurl.replace("_interval=%s" % gran, "_interval=day")
        dayrequest = requests.get(stuffurl, headers=authdict)
    else:
        dayrequest = request

    # figure out the last event date
    result = json.loads(dayrequest.text)
    lasteventdate = max([i[0] for i in result['ts']])

    result = json.loads(request.text)

    return result, lasteventdate


def process_event_date(query, authdict, gran,req_date):
    print("Processing an event")
    stuffurl = "http://sage-dev-internal.isi.edu/mydig/projects/%s/search/event?%s&event_date$less-equal-than=%s" % (PROJECT_NAME, query,req_date)
    print(stuffurl)

    request = requests.get(stuffurl, headers=authdict)
    if request.status_code == 404:
        return None,None

    # if the granularity is not daily, then figure out the last *day* with an event
    if gran != 'day':
        stuffurl = stuffurl.replace("_interval=%s" % gran, "_interval=day")
        dayrequest = requests.get(stuffurl, headers=authdict)
    else:
        dayrequest = request

    # figure out the last event date
    result = json.loads(dayrequest.text)
    lasteventdate = max([i[0] for i in result['ts']])

    result = json.loads(request.text)

    return result, lasteventdate

granularity = re.compile(r".*_interval=(?P<interval>\w+)\b.*")
offset = re.compile(r".*_offset=(?P<offset>\w+)\b.*")

def n_in_array(x, array):
    for i, v in enumerate(array):
        if x >= v:
            return i

def update_charts():
    # establish the template bank and query manipulation thingy
    bank = IFPTemplateBank("templates.txt")
    qm = IFPKGToQuery()

    # figure out how many IFPs there are
    authdict = {"authorization": "Basic ZGlnOmRJZ0RpRw=="}
    request = requests.get("http://sage-dev-internal.isi.edu/mydig/projects/%s/search/conjunctive?type=ifp"
                           % (PROJECT_NAME,),
                           headers=authdict)
    result = json.loads(request.text)
    num_ifps = result['hits']['total']

    # get all of them
    request = requests.get(
        "http://sage-dev-internal.isi.edu/mydig/projects/%s/search/conjunctive?type/key=ifp&_size=%d"
        % (PROJECT_NAME, num_ifps),
        headers=authdict)
    kg_entries = json.loads(request.text)

    search_key = "".lower()  # This will search for ALL IFPs

    id_list_req = requests.post('https://sage-rct.isi.edu/api/ql', data='{"query":"query{questions(limit:999999){hfcId}}","variables":"null"}', headers={'Content-Type': 'application/json'})
    id_set = frozenset([e['hfcId'] for e in id_list_req.json()['data']['questions']])

    black_id_set = frozenset([])
    for doc in kg_entries['hits']['hits']:
        # if the IFP hasn't resolved, we havec no way to score it
        if not doc['_source']['ifp']['resolved?']:
            continue

        if not int(doc['_source']['ifp']['id']) in id_set:
            continue

        if int(doc['_source']['ifp']['id']) in black_id_set:
            continue

        # find the most similar templates
        if doc['_source']['ifp']['name'].lower().find(search_key) == -1:
            continue

        results = bank.most_similar_to_kg_entry(doc, "ifp__name", topn=1)
        matched_sentence, matched_result, replacements, score = results[0]

        # two things that keep getting jacked up: FAO prices and oil, make sure they actually appear in all questions
        if (" FAO " in matched_sentence and " FAO " not in doc['_source']['ifp']['name'])\
                or (" crude oil " in matched_sentence and " crude oil " not in doc['_source']['ifp']['name']) \
                or (" loya jirga " in matched_sentence.lower() and " loya jirga " not in doc['_source']['ifp']['name'].lower()):
            score += MAX_SIMILARITY_SCORE + 1

        #print("-- IFP --")
        #print("\tIFP:", doc['_source']['ifp']['name'].lower())
        #print("\tMatched:", matched_sentence)
        #print("\tTHE SCORE IS:", score)

        if score > MAX_SIMILARITY_SCORE:
            #print("\tGross, the score was %f" % score)
            continue

        update_days = None
        print("Found an ifp!!! ", doc['_source']['ifp']['id'])
        if 'last_event_date_key' in matched_result:
            dataset_name = matched_result['last_event_date_key']
            dataset_json_file = 'datasets/{}/{}_dataset.json'.format(dataset_name, dataset_name)
            if os.path.exists(dataset_json_file):
                dataset_json = json.loads(open(dataset_json_file).read())
                frequency = dataset_json['where_to_download']['frequency']
                if 'dayofmonth:' in frequency:
                    update_days = [int(d) for d in frequency.replace('dayofmonth:','').split(',')]
                    print('update_days:', update_days)
                    assert len(update_days) == 1

        assert len(matched_result['queries']) == 1
        for q in matched_result['queries']:
            # take the mapped query and fill it in with actual values.
            mapped_queries = qm.fill_query(q, replacements)
            assert len(mapped_queries) == 1

            for querydoc in mapped_queries:
                query = querydoc['query']
                shouldFlip = querydoc['flip']
                # pull the granularity out of the selected query
                if 'granularity_override' in matched_result:
                    gran = matched_result['granularity_override']
                else:
                    granmatch = granularity.match(query)
                    gran = "month"
                    if granmatch is not None:
                        gran = granmatch.group('interval')

                offmatch = offset.match(query)
                if offmatch:
                    offs = offmatch.group('offset')
                else:
                    offs = '0d'

                # fetch results
                print("Query is", query)
                lasteventdate = None
                if matched_result['query_type'] == 'timeseries':
                    result = process_timeseries(query, authdict, gran, offs)
                elif matched_result['query_type'] == 'event':
                    result, lasteventdate = process_event(query, authdict, gran)

                # here is where we test if we got data. if we did, then we may need to flip
                if result is not None:
                    # hand it to andy
                    cvt, shouldrun = toandy(result, doc['_source']['ifp'], gran, lasteventdate, shouldFlip,
                                            fake_a_date=False)
                    ifpid = cvt['ifp']['id']

                    if shouldrun == "Valid":
                        if 'do_not_forecast' in matched_result and matched_result['do_not_forecast'] == True:
                            pass
                        else:
                            cur_date = cvt['ifp']['starts_at'].split('T')[0]
                            end_date = cvt['ifp']['ends_at'].split('T')[0]

                            cur_date = datetime.strptime(cur_date, "%Y-%m-%d")
                            end_date = datetime.strptime(end_date, "%Y-%m-%d")

                            if end_date >= datetime.now():
                                continue

                            with open('data_all/andy_input_%d.json' % ifpid, 'w') as fout:
                                json.dump(cvt, fout)

                            i=0
                            while cur_date <= end_date:
                                save_date = cur_date.strftime("%Y-%m-%d")
                                req_date = cur_date.strftime("%Y-%m-%d")

                                if update_days is not None:
                                    if cur_date.day < update_days[0]:
                                        data_date = cur_date - relativedelta(days=cur_date.day)
                                        req_date = data_date.strftime("%Y-%m-%d")

                                print(req_date, save_date)
                                if matched_result['query_type'] == 'timeseries':
                                    result, lasteventdate = process_timeseries_date(query, authdict, gran, offs,req_date)
                                elif matched_result['query_type'] == 'event':
                                    result, lasteventdate = process_event_date(query, authdict, gran,req_date)

                                cvt, shouldrun = toandy(result, doc['_source']['ifp'], gran, lasteventdate, shouldFlip,
                                            fake_a_date=False)

                                with open('data_backcast/andy_input_{}_{}.json'.format(ifpid, save_date), 'w') as fout:
                                    json.dump(cvt, fout)

                                def post_server(_save_date, _cvt, _ifpid):
                                    # no need to specify drop_after
                                    andy_response = requests.post("http://sage-rct.isi.edu:6002/forecast?quick=False", json=_cvt)
                                    result = json.loads(andy_response.text)
                                    output_filename = 'data_output/andy_output_{}_{}.json'.format(_ifpid, _save_date)
                                    print('Done', output_filename)
                                    with open(output_filename, 'w') as fout:
                                        json.dump(result, fout)

                                executor.submit(post_server, save_date, cvt, ifpid)
                                i += 1
                                cur_date += timedelta(days=1)

    print('All task has been generated')
    executor.shutdown()
    print('All done')

if __name__ == "__main__":

    update_charts()
