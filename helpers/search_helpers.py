# import helpers.general_helpers
from helpers.general_helpers import *
import numpy as np


def binary_ranking(index, postings_path, query):
    index_terms = index.term_total.keys()
    query_tokens = tokenize_removeStopWord(query)
    unique_tokens = np.unique(query_tokens)
    relevant_query_tokens = filter(lambda term: term in index_terms, unique_tokens)
    doc_id_freq_count = {}

    for term in unique_tokens:
        posting_list = index.read_posting_list(term, postings_path)

        for doc_id, doc_freq in posting_list:
            doc_id_freq_count[doc_id] = doc_id_freq_count.get(doc_id, 0)

    FREQ_COUNT_INDEX = 1
    sorted_by_frequency = sorted(doc_id_freq_count.items(), key=lambda doc_frq_c: doc_frq_c[FREQ_COUNT_INDEX], reverse=True)

    return sorted_by_frequency
