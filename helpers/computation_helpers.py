import math
from collections import Counter


from helpers.general_helpers import tokenize_removeStopWord
from helpers.general_helpers import get_doc_id_title_list

doc_avg_mean = 320
number_of_docs = 6348910


def cosine_Similarity_calc(query, index, path):
    query_vector = {}
    for term in query:
        if term in query_vector:
            query_vector[term] += 1
        else:
            query_vector[term] = 1

    scores = {}
    for term in query_vector:
        posting_list = index.read_posting_list(term, path)
        for doc_id, tf in posting_list:
            if doc_id in scores:
                scores[doc_id] += tf * query_vector[term] * math.log(number_of_docs / index.df[term], 10)
            else:
                scores[doc_id] = tf * query_vector[term] * math.log(number_of_docs / index.df[term], 10)

    for doc_id in scores:
        scores[doc_id] = (scores[doc_id] / (
                doc_avg_mean * math.sqrt(sum([x ** 2 for x in query_vector.values()]))))

    return scores


def cosine_Similarity(query, index, path, doc_id_len):
    query_vector = {}
    for term in query:
        if term in query_vector:
            query_vector[term] += 1
        else:
            query_vector[term] = 1

    scores = {}
    for term in query_vector:
        posting_list = index.read_posting_list(term, path)
        for doc_id, tf in posting_list:
            if doc_id in scores:
                scores[doc_id] += tf * query_vector[term] * math.log(number_of_docs / index.df[term], 10)
            else:
                scores[doc_id] = tf * query_vector[term] * math.log(number_of_docs / index.df[term], 10)

    for doc_id in scores:
        scores[doc_id] = scores[doc_id] / (
                (int(doc_id_len[str(doc_id)])) * math.sqrt(sum([x ** 2 for x in query_vector.values()])))

    return scores


def calc_idf(query):
    # calculate bm25 by the formula we learn at the class
    idf = {}
    for term in query:
        if term not in body_index.term_total.keys():
            idf[term] = 0
        else:

            mone = number_of_docs - body_index.df[term] + 0.5
            mechane = body_index.df[term] + 0.5 + 1
            idf[term] = math.log((mone / mechane) + 1)

    return idf
