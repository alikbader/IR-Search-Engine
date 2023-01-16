# import helpers.general_helpers
from helpers.general_helpers import *
import numpy as np
import math

number_of_docs = 6348910
doc_avg_mean = 320


def binary_ranking(index, postings_path, query, stemming=False):
    index_terms = index.df.keys()
    query_tokens = tokenize_removeStopWord(query)
    unique_tokens = np.unique(query_tokens)
    relevant_query_tokens = list(filter(lambda term: term in index_terms, unique_tokens))
    stemmed_tokens = stem_tokens(query_tokens)
    combined_tokens = []

    for token in stemmed_tokens:
        combined_tokens.append(token)

    for token in relevant_query_tokens:
        combined_tokens.append(token)

    combined_tokens = np.unique(combined_tokens)

    if stemming:
        relevant_query_tokens = combined_tokens

    doc_id_freq_count = {}

    for term in relevant_query_tokens:
        # if term in index_terms:
        posting_list = index.read_posting_list(term, postings_path)

        for doc_id, doc_freq in posting_list:
            doc_id_freq_count[doc_id] = doc_id_freq_count.get(doc_id, 0)

    FREQ_COUNT_INDEX = 1
    sorted_by_frequency = sorted(doc_id_freq_count.items(), key=lambda doc_frq_c: doc_frq_c[FREQ_COUNT_INDEX],
                                 reverse=True)

    return sorted_by_frequency


def BM25(doc_id_len, query, index, path, k1=1.2, b=0.75):
    query_vector = {}
    for term in query:
        if term in query_vector:
            query_vector[term] += 1
        else:
            query_vector[term] = 1

    scores = {}
    for term in query_vector:
        if term in index.df:
            posting_list = index.read_posting_list(term, path)
            for doc_id, tf in posting_list:
                idf = math.log((number_of_docs - index.df[term] + 0.5) / (index.df[term] + 0.5))
                doc_length = int(doc_id_len.get(str(doc_id), str(doc_avg_mean)))
                score = (idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * (doc_length / doc_avg_mean))))
                if doc_id in scores:
                    scores[doc_id] += score
                else:
                    scores[doc_id] = score

    return scores
