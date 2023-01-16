# IR-Search-Engine
# wikipedia search engine

## Intro:
We are students in our fourth (Ali) and third (Ilan) in Ben Gurion university, during the course of Information Retrival we created this project, in which we
return our best resutls based on a given query, with different type of methods. We used several types of technologies and and different libraries in Python.

## The Project:
### First steps:
- We started by creating 3 **posting lists**:
  - body_postings
  - title_postings
  - anchor_postings
- Then we created an Inverted index for each one of them.
- saved in csv file each doc id with it's length.
- saved in another csv file the doc id and the title (in order to read it later to dict and retrive the information faster).

* We have also tried to create csv files for each term, and the **tfidf** in each document, but the files were to big, so we decided to calculate by ourselves during the function.
* We have also tried to calculate the norma for the cosine similarity, but because of the reason abouve we decided to leave it.

### Calculation functions:
We have 3 different types of calculation functions:
- **Binary search** - which is done on the body (in search_body) and anchor (in search_anchor) - We search if the word is in the document and if it is than how many times,
  each time it appears we add one.
- **BM25** - We have used this method only in the main search- this function calculates the similarity between term and relevant doc we the BM25 function.
- **Cosine Similarity** - We have used this method only in the body search- this function calculates the similarity between term and relevant doc we the cosine similarity function.

### Search functions:
We have 6 endpoints:
- **Get: Page_view** : Returns the number of page views that each of the provide wiki articles.
- **Get: Page_rank** : Returns PageRank values for a list of provided wiki article IDs.
- **Seach_anchor** : Returns all search results that contain a query word in the title text of articles, ordered in descending order of the
                     number of query words that appear in anchor text linking to the page.
- **Serch_title** : Returns all search results that contain a query word in the title text of articles, ordered in descending order of the
                    number of query words that appear in anchor text linking to the page.
- **Search_body** : Returns up to a 100 search results for the query using tfidf and cosine similarity of the body of articles only.
- **Search** : The main search function which we implemented how we wanted, after trying couple of methods and training our model we chose the BM25 with K=1.7 B=0.7.

### Results:
- Our precision  **0.419** and the Average Time for query **2.41 seconds**.
![image](https://user-images.githubusercontent.com/103121260/212740143-4efdf95b-fa01-42df-8071-08cc33a4542e.png)
