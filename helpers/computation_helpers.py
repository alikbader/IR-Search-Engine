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


def BM25_score_doc_id_query(avg_doc_length_of_all_corpus, term_docid_freq, query, doc_id, k1=1.5, b=0.75):
    # In this function for each term in the query and to each doc that relevantive to the query (that term in this document)
    # We calculate the bm25 score between the document to all the terms in the query!

    idf = calc_idf(query)
    bm25 = 0
    for term in query:

        # by the formula like in homework 4 !
        if (term, doc_id) in term_docid_freq:
            freq = term_docid_freq[(term, doc_id)]
            first = (k1 + 1) * freq
            secondpart = query[term] * idf[term]
            thirdpart = freq + k1 * (1 - b + b * (int(doc_id_len[str(doc_id)]) / avg_doc_length_of_all_corpus))

            bm25 += secondpart * (first / thirdpart)
    return bm25


def BM25_search_body3(queries, N=100):
    # aveage doc length
    avg_doc_length_of_all_corpus = sum(body_index.DL.values()) / len(body_index.DL)

    # size of the corpus
    size_corpus = len(body_index.DL)

    tokens = tokenize_removeStopWord(queries)
    all_docs_distinct = []
    term_docid_freq = {}

    # for on all the terms in the query
    for term in tokens:

        if term in body_index.term_total:

            list_docid_tf_foreach_term = body_index.read_posting_list(term, extended_body_index_path)
            lst_docid = []

            # getting a list of doc id to each term
            # getting a dictionary that to each (doc_id,term) his tf-term frequency
            for doc_id, freq in list_docid_tf_foreach_term:
                term_docid_freq[(term, doc_id)] = freq
                lst_docid.append(doc_id)

            all_docs_distinct += lst_docid

    # getting only distinct docs
    all_docs_distinct = set(all_docs_distinct)

    doc_id_bm25 = []
    for doc_id in all_docs_distinct:
        doc_id_bm25.append(
            (doc_id,
             BM25_score_doc_id_query(avg_doc_length_of_all_corpus, term_docid_freq, dict(Counter(tokens)), doc_id, 1.5,
                                     0.75, )))

    doc_id_bm25 = sorted(doc_id_bm25, key=lambda x: x[1], reverse=True)[:100]

    res = get_doc_id_title_list(doc_id_bm25)

    return res
