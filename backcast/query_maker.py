import json
from datetime import datetime, timedelta

epoch_date = datetime.strptime('1970-01-01T00:00:00', "%Y-%m-%dT%H:%M:%S")


class IFPKGToQuery(object):
    """
    This is a class to replace tokens in a knowledge graph entry.
    """

    def __init__(self, mapping_dict=None):
        """
        Initialize the class!
        :param mapping_dict:
        The dictionary of keys to look for and their names.
        """
        if mapping_dict is None:
            self.mapping_dict = {
                "country": "country",
                "region": "region",
                "magnitude": "magnitude",
                "item": "item",
                "date_mentioned_in_question": "date",
                "month_year_mentioned_in_question": "date",
                "currency": "currency"
            }
        else:
            self.mapping_dict = mapping_dict
        pass

    def construct_query(self, kg_entry, segment='ifp__name', keyset=set([])):
        """
        This will build the query from the knowledge graph entry. Specifically, it will replace all instances
        of the keys_to_replace with query parameters.
        E.g., "Will ACLED record any civilian fatalities in Angola in October 2017?", will be mapped to
        "Will ACLED record any civilian fatalities in __country in __date?"
        :param kg_entry:
        The knowledge graph entry as a Python document.
        :param segment:
        The segment of the KG entry to assess.
        :return:
        A dictionary containing:
        tokens -> The tokens of the transformed sentence.
        ogvalues -> A list from the keys to the original values
        """
        tokens = kg_entry['_source']['content_extraction'][segment]
        if type(tokens) is list:
            tokens = tokens[0]
        tokens = tokens['simple_tokens_original_case']
        extractions = kg_entry['_source']['knowledge_graph']

        all_replacements = []
        all_original_values = []
        for key, keytoreplace in self.mapping_dict.items():
            if key not in extractions or keytoreplace not in keyset:
                continue
            for replacements in extractions[key]:
                for replacement in replacements['provenance']:
                    ev = replacement['extracted_value']
                    all_original_values.append(("__%s" % keytoreplace, replacements['value']))
                    # if replacement['source']['segment'] != segment:
                    #     continue
                    start_idx = replacement['source']['context']['start']
                    end_idx = replacement['source']['context']['end']
                    all_replacements.append((start_idx, end_idx, "__%s" % keytoreplace))

        # perform the replacements in reverse order of their appearance so that indexes don't get screwed up
        all_replacements = sorted(all_replacements, key=lambda x: x[0], reverse=True)
        for start, end, replace in all_replacements:
            tokens = tokens[0:start] + [replace] + tokens[end:]

        return {
            "tokens": tokens,
            "key2og": all_original_values
        }

    def fill_query(self, query_string, values, mutables='__exchange_rate_from_currency'):
        """
        This function fills out a query
        :param query_string:
        The string of the DIG Query.
        :param values:
        The list of (field, value) to replace. It will be consumed in order. No checks will be made to ensure
        that all of the keys are used.
        :param mutables:
        A key that, when present, will be permuted. This code assumes that each of these mutables will be
        used at most 2 times
        :return:
        The query replaced with the values
        """
        mutablesfound = []

        for f, v in values:
            if f == mutables:
                mutablesfound.append(v)
            else:
                f = "{%s}" % f.replace("__", "")
                query_string = query_string.replace(f, v, 1)

        qs = []
        if len(mutablesfound) > 1:
            f = "{%s}" % mutables.replace("__", "")
            for mpair, flip in [(mutablesfound, False), (mutablesfound.reverse(), True)]:
                tempq = query_string
                for i in mpair:
                    tempq = tempq.replace(f, i, 1)
                qs.append({
                    "query": tempq,
                    "flip": flip
                })
        elif mutablesfound == 1:
            f = "{%s}" % mutables.replace("__", "")
            qs.append({
                "query": query_string.replace(f, mutables[0]),
                "flip": False
            })
        else:
            qs.append({
                "query": query_string,
                "flip": False
            })

        try:
            # for each query, if it contains an {interval} and two __dates, then replace the interval with the difference.
            alldates = [j for i, j in values if i == '__date']
            alldates = [datetime.strptime(i, "%Y-%m-%dT%H:%M:%S") for i in alldates]
            interval = calculate_interval(alldates)
        except ValueError as ve:
            interval = None
            alldates = []

        if interval:
            f_offset = find_offset(alldates, interval)
            offset = f_offset if f_offset else '0d'
            interval = "%dd" % (interval)
            for qidx in range(len(qs)):
                qs[qidx]['query'] = qs[qidx]['query'].replace("{interval}", interval).replace("{offset}", offset)

        return qs


def find_offset(alldates, interval):
    if len(alldates) > 1:
        end_date = max(alldates)
        delta = (end_date - epoch_date).days
        offset = delta % interval
        return "%dd" % offset
    return None


def calculate_interval(alldates):
    if len(alldates) > 1:
        interval = max(alldates) - min(alldates)
        return interval.days + 1
    return None


if __name__ == "__main__":
    q = IFPKGToQuery()
    doc = json.loads(open("example.json").read())
    toks = q.construct_query(doc)
    print(toks)
