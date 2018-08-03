import json
import numpy as np
from nltk import word_tokenize
from query_maker import IFPKGToQuery


def is_special(x):
    """
    This is the function to determine whether a string is a special word. At this time, a special word is
    anything beginning with "__"
    :param x: The string to test for specialness.
    :return: Whether it's special.
    """
    return len(x) >= 2 and x[0:2] == "__"


def weighted_lev(sourceseq, destseq, sagepenalty=100, softsub=True):
    """
    This will compute a Levenshtein distance between the source sequence (sourceseq) and the
    destination sequence (destseq). The special twist here is that our IFPs have some special words. If these special
    words need to be deleted, inserted, or substituted, then we will apply a LARGE penalty (sagepenalty) to the
    distance.

    This function can also handle typos between the words. Typos are only considered for non-special words. In the
    event of a typo, the penalty will be the levenshtein distance between those words.

    :param sourceseq: The source sequence. Can be an array of strings, or strings themselves.
    :param destseq: The destination sequence. Can be an array of strings, or strings themselves.
    :param sagepenalty: The penalty applied to perturbations on special words. Should be a large positive number.
    :param softsub: A boolean indicating whether or not to perform soft substitution penalties for typos.
        If set to true, a typo will be penalized according to the normalized levenshtein distance of the words.
    :return:
        The weighted levenshtein distance.
    """

    #  This is the matrix that will accrue the differences between the strings.
    #  As an optimization measure, we can use just two vectors. Left for "Future work."
    mtx = np.zeros((len(sourceseq) + 1, len(destseq) + 1))

    #  We can transform the source prefixes by deleting every element. This becomes the first column.
    for i in range(1, len(sourceseq) + 1):
        penalty = sagepenalty if is_special(sourceseq[i - 1]) else 1
        mtx[i, 0] = mtx[i - 1, 0] + penalty

    # Similarly, we can reach the target simply by inserting every element. This becomes the first row.
    for j in range(1, len(destseq) + 1):
        penalty = sagepenalty if is_special(destseq[j - 1]) else 1
        mtx[0, j] = mtx[0, j - 1] + penalty

    # Iterate outward over each source, target to find the cheapest path.
    for j in range(1, len(destseq) + 1):
        for i in range(1, len(sourceseq) + 1):
            # determine the cost of a substitution.
            # if the strings are THE SAME, it should be 0.
            if sourceseq[i - 1] == destseq[j - 1]:
                subcost = 0
            else:
                # if the left or right is a flagged field,
                if is_special(sourceseq[i - 1]) or is_special(destseq[j - 1]):
                    subcost = sagepenalty
                else:
                    if softsub:
                        subcost = weighted_lev(sourceseq[i - 1], destseq[j - 1], 1, False)
                        # normalize the sub-value by the longest possible value
                        mv = max(len(sourceseq[i - 1]), len(destseq[j - 1]))
                        subcost /= mv
                    else:
                        subcost = 1

            leftpen = sagepenalty if is_special(sourceseq[i - 1]) else 1
            rightpen = sagepenalty if is_special(destseq[j - 1]) else 1

            mtx[i, j] = min(
                mtx[i - 1, j] + leftpen,  # deletion
                mtx[i, j - 1] + rightpen,  # insertion
                mtx[i - 1, j - 1] + subcost  # substitution
            )

    # the final, best cost will be the bottom-right value in the matrix.
    return mtx[len(sourceseq), len(destseq)]


class IFPTemplateBank(object):
    """
    This class holds a bank of IFP templates and measures the similarity to other templates.
    """

    def __init__(self, bankfile, sagepenalty=100, use_softsub=True):
        """
        This builds the bank of IFPs
        :param bankfile:
            The string location of the bank. It assumes each line is a string of the template.
            TODO: Make this take a JSON.
        :param sagepenalty:
            The large integer value to set special word penalties to.
        :param use_softsub:
            The boolean indicating whether we penalize typos.
        """
        self.templates = []
        self.sagepenalty = sagepenalty
        self.use_softsub = use_softsub
        self.qm = IFPKGToQuery()

        with open(bankfile) as bf:
            for line in bf:
                doc = json.loads(line)
                keyset = doc['keyset']
                for t in doc['expanded_tempates']:
                    # store the template and the doc
                    self.templates.append({
                        "tokens": word_tokenize(t),
                        "sentence": t,
                        "document": doc,
                        "keyset": keyset
                    })

    @staticmethod
    def __rank_and_return(scored, topn, return_scores, sortkey=2):
        """
        This ranks and returns a list of templates.
        :param scored:
        The scored (sentence, document, score) tuples.
        :param topn:
        The n best to return
        :param return_scores:
        Whether or not to return the scores.
        :param sortkey:
        The index of the key to sort.
        :return:
        The sorted list. Trimmed and fields filtered depending on input.
        """
        scored = sorted(scored, key=lambda x: x[sortkey])
        if topn > 0:
            scored = scored[0:topn]

        if not return_scores:
            scored = [i[0] for i in scored]

        return scored

    def most_similar_templates(self, sentence, topn=-1, return_scores=True):
        """
        Returns the k most similar templates to a provided sentence
        :param sentence: The sentence to test.
        :param topn: The n most similar templates to return. Should be a positive integer, with the exception of -1,
        in which case all will be returned.
        :param return_scores:
            whether or not to return the distance score.
        :return:
            A list of (topn) tuples containing (the raw sentence, the entry in the template dictionary, [the distance score]).
        """
        scored = []

        stemp = word_tokenize(sentence)
        for t in self.templates:
            target_sentence = t['tokens']
            raw_sentence = t['sentence']
            doc = t['document']
            scored.append((raw_sentence, doc, weighted_lev(stemp, target_sentence, self.sagepenalty, self.use_softsub)))

        return self.__rank_and_return(scored, topn, return_scores)

    def most_similar_to_kg_entry(self, kgentry, kgsegment, topn=-1, return_scores=True):
        """
        Find the templates best suited for this template.
        :param kgentry:
        The knowledge graph entry as a dictionary.
        :param kgsegment:
        The segment to check (this is where we look for the tokens).
        :param topn: The n most similar templates to return. Should be a positive integer, with the exception of -1,
        in which case all will be returned.
        :param return_scores:
            whether or not to return the distance score.
        :return:
            A list of (topn) tuples containing
                (the raw sentence, the entry in the template dictionary,
                the dictionary of fill values for the match, [the distance score])
        """
        scored = []

        # print("Testing question: ", kgentry['_source']['ifp']['name'])
        for t in self.templates:
            stemp = self.qm.construct_query(kgentry, kgsegment, t['keyset'])
            tokens = stemp['tokens']
            # print("These are the tokens", stemp)
            target_sentence = t['tokens']
            raw_sentence = t['sentence']
            doc = t['document']
            scored.append(
                (raw_sentence, doc, stemp['key2og'],
                 weighted_lev(tokens, target_sentence, self.sagepenalty, self.use_softsub))
            )

        return self.__rank_and_return(scored, topn, return_scores, sortkey=3)


if __name__ == "__main__":
    bank = IFPTemplateBank("mahboobehs_templates.txt")
    # for template, doc, score in bank.most_similar_templates("__country between __date and __date ?"):
    s = "Will the PITF Worldwide Atrocities Dataset record an event perpetrated by a non-state actor in " \
        "__country that starts between __date and __date?"
    for template, doc, score in \
            bank.most_similar_templates(s):
        print(template, score)
