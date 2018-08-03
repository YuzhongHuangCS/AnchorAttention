import json
import re
import requests
import sys
from ifp_similarity_bank import IFPTemplateBank
from query_maker import IFPKGToQuery
from toandy import toandy
from fromandy import andyzipper
from requests.auth import HTTPBasicAuth
from slack_client import SlackClient
from datetime import datetime, timedelta
import time
from time import sleep


if len(sys.argv) < 3:
    # print("run_pipeline.py PROJECT_NAME MAX_SIMILARITY_SCORE")
    # sys.exit(1)
    PROJECT_NAME = "sage_kg_v2"
    MAX_SIMILARITY_SCORE = 8.0
else:
    PROJECT_NAME = sys.argv[1]
    MAX_SIMILARITY_SCORE = float(sys.argv[2])

sc = SlackClient('xoxp-62576775760-239663949941-334692921798-9722db4eb5c732780ad8da973029d1fc')
# sc.chat_post_message('#sage-support', 'Hey this is Fred testing the API thingy.',
#                      username='SAGE Dataset Updater')


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


def process_event(query, authdict, gran):
    print("Processing an event")
    stuffurl = "http://sage-dev-internal.isi.edu/mydig/projects/%s/search/event?%s" % (PROJECT_NAME, query)
    print(stuffurl)

    request = requests.get(stuffurl, headers=authdict)
    if request.status_code == 404:
        return None

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

    search_key = "Loya Jirga".lower()  # This will search for ALL IFPs

    for doc in kg_entries['hits']['hits']:
        # find the most similar templates
        if doc['_source']['ifp']['name'].lower().find(search_key) == -1:
            continue
        print("Checking out IFP:", doc['_source']['ifp']['name'])

        start = time.time()
        results = bank.most_similar_to_kg_entry(doc, "ifp__name", topn=1)

        matched_sentence, matched_result, replacements, score = results[0]

        # two things that keep getting jacked up: FAO prices and oil, make sure they actually appear in all questions
        if (" FAO " in matched_sentence and " FAO " not in doc['_source']['ifp']['name'])\
                or (" crude oil " in matched_sentence and " crude oil " not in doc['_source']['ifp']['name']) \
                or (" loya jirga " in matched_sentence.lower() and " loya jirga " not in doc['_source']['ifp']['name'].lower()):
            score += MAX_SIMILARITY_SCORE + 1

        print("-- IFP --")
        print("\tIFP:", doc['_source']['ifp']['name'].lower())
        print("\tMatched:", matched_sentence)
        print("\tTHE SCORE IS:", score)

        if score > MAX_SIMILARITY_SCORE:
            print("\tGross, the score was %f" % score)
            continue

        print("Found an ifp!!! ", doc['_source']['ifp']['id'])

        end = time.time()
        print("Took %0.1f seconds." % (end - start))

        for q in matched_result['queries']:
            # take the mapped query and fill it in with actual values.
            mapped_queries = qm.fill_query(q, replacements)

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
                        print(json.dumps(cvt))
                        if 'do_not_forecast' in matched_result and matched_result['do_not_forecast'] == True:
                            cvt = andyzipper(cvt, None)
                        else:
                            andy_response = requests.post("http://sage-rct.isi.edu:5000/forecast", json=cvt)
                            print("ANDY")
                            print(andy_response.text)
                            open("andyio/andy_input_%d.json" % ifpid, "w").write(json.dumps(cvt))
                            open("andyio/andy_output_%d.json" % ifpid, "w").write(andy_response.text)

                            cvt = andyzipper(cvt, json.loads(andy_response.text))

                        cvt['ylabel'] = matched_result['ylabel'] if 'ylabel' in matched_result else 'Value'

                        print("WEEEEEE")
                        response = requests.put(
                            "http://sage-dev-internal.isi.edu/es/ifp_predictions/predictions/%d" %
                            doc['_source']['ifp']['id'],
                            auth=HTTPBasicAuth('dig', 'dIgDiG'), json=cvt)
                        print(json.dumps(cvt))
                        print(response.content)
                        print("Wrote it to:", "http://sage-dev-internal.isi.edu/es/ifp_predictions/predictions/%d" %
                              doc['_source']['ifp']['id'])


if __name__ == "__main__":
    while True:
        # sleep until 10PM
        t = datetime.today()
        future = datetime(t.year, t.month, t.day, 22, 0)
        if t.hour >= 22:
            future += timedelta(days=1)
        update_charts()
        print("Waiting until 10pm. %d seconds from now." % int((future - t).seconds))
        sleep((future - t).seconds)
