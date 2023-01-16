import csv

import numpy as np
from flask import Flask, request, jsonify

from inverted_index_gcp import *

from helpers import *
import helpers.computation_helpers
import helpers.general_helpers
import helpers.search_helpers

from helpers.computation_helpers import *
from helpers.general_helpers import *
from helpers.search_helpers import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd

import gzip


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# region Setup

# read inverted indexes
title_index_path = 'postings/title_postings/postings_gcp'
body_index_path = 'postings/body_postings/postings_gcp'
anchor_index_path = 'postings/anchor_postings/postings_gcp'

extended_title_index_path = 'postings/title_postings/postings_gcp/'
extended_body_index_path = 'postings/body_postings/postings_gcp/'
extended_anchor_index_path = 'postings/anchor_postings/postings_gcp/'

title_index_name = 'index'
body_index_name = 'index'
anchor_index_name = 'index'

title_index = InvertedIndex.read_index(title_index_path, title_index_name)
body_index = InvertedIndex.read_index(body_index_path, body_index_name)
anchor_index = InvertedIndex.read_index(anchor_index_path, anchor_index_name)
# read doc_id_title
doc_id_title_name = 'doc_id_title.csv'
doc_id_title = read_csv_to_dict(doc_id_title_name)
# read doc_id_len
doc_id_len_name = 'doc_id_len.csv'
doc_id_len = read_csv_to_dict(doc_id_len_name)

doc_avg_mean = 320
number_of_docs = 6348910

print('read dic id len')
# read page rank
page_rank_path = 'page_rank.csv'
page_rank = read_csv_to_dict(page_rank_path)
# read page views
page_views_path = 'pageviews-202108-user.pkl'
page_views = pd.read_pickle(page_views_path)


# endregion Setup

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    query_tokens = tokenize_removeStopWord(query)
    query_tokens = query_tokens

    unique_tokens = np.unique(query_tokens)

    k = 1.7
    b = 0.7

    if len(query_tokens) == 0:
        return jsonify(res)

    if len(query_tokens) > 2:
        combined_scores = BM25(doc_id_len, query_tokens, body_index, body_index_path, k, b)
    else:
        if len(query_tokens) == 1:
            body_weight = 1
            title_weight = 0
            anchor_weight = 0

        if len(query_tokens) == 2:
            body_weight = 0.7
            title_weight = 0.2
            anchor_weight = 0.1

        res_body = BM25(doc_id_len, query_tokens, body_index, body_index_path, k, b)
        res_anchor = cosine_Similarity_calc(unique_tokens, anchor_index, anchor_index_path)
        res_title = cosine_Similarity_calc(unique_tokens, title_index, title_index_path)

        combined_scores = defaultdict(float)
        for doc_id, score in res_body.items():
            combined_scores[doc_id] += score * body_weight
        for doc_id, score in res_title.items():
            combined_scores[doc_id] += score * title_weight
        for doc_id, score in res_anchor.items():
            combined_scores[doc_id] += score * anchor_weight

    res = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:10000]
    res = sorted(res, key=lambda x: page_views[str(x[0])], reverse=True)[:1000]
    res = sorted(res, key=lambda x: x[1], reverse=True)[:100]

    res = get_doc_id_title_list(res, doc_id_title)

    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    query_tokens = tokenize_removeStopWord(query)
    unique_tokens = np.unique(query_tokens)

    res = cosine_Similarity(unique_tokens, body_index, body_index_path, doc_id_len)
    res = sorted(res.items(), key=lambda x: x[1], reverse=True)[:100]
    res = get_doc_id_title_list(res, doc_id_title)

    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    binary_ranking_results = binary_ranking(title_index, title_index_path, query)
    res = get_doc_id_title_list(binary_ranking_results, doc_id_title)

    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')

    if len(query) == 0:
        return jsonify(res)

    binary_ranking_results = binary_ranking(anchor_index, extended_anchor_index_path, query)
    res = get_doc_id_title_list(binary_ranking_results, doc_id_title)

    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    print(wiki_ids)
    if len(wiki_ids) == 0:
        return jsonify(res)

    for page_id in wiki_ids:
        if str(page_id) in page_rank.keys():
            res.append(page_rank[str(page_id)])

    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)

    page_views_res = []

    for wiki_id in wiki_ids:
        wiki_page_views = page_views.get(wiki_id, None)
        page_views_res.append(wiki_page_views)

    return jsonify(page_views_res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
