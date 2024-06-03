from fastapi import FastAPI
from pydantic import BaseModel
from elasticsearch import Elasticsearch
from datetime import datetime
import csv
import nltk
from nltk.corpus import wordnet as wn
import spacy
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["http://localhost:5173"]

app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

es = Elasticsearch(
    ['http://127.0.0.1:9200'],
    request_timeout=300  
)
nlp = spacy.load("en_core_web_sm")
class QueryRequest(BaseModel):
    search_text: str
@app.post("/autocomplete/")
def autocomplete(request: QueryRequest):
    search_text = request.search_text
    
    expanded_terms = expand_query_terms(search_text)

    suggest_response = es.search(
        index='autocomplete',
        body={
            'suggest': {
                'query-suggest': {
                    'prefix': search_text,
                    'completion': {
                        'field': 'query',
                        'size': 100  
                    }
                }
            }
        }
    )

    match_response = es.search(
        index='autocomplete',
        body={
            'query': {
                'bool': {
                    'should': [
                        {'match': {'query_text': term}} for term in expanded_terms
                    ]
                }
            },
            'size': 100  
        }
    )

   
    suggest_suggestions = suggest_response['suggest']['query-suggest'][0]['options']
    match_hits = match_response['hits']['hits']

    suggest_texts = [suggestion['text'] for suggestion in suggest_suggestions]
    match_texts = [hit['_source']['query_text'] for hit in match_hits]

    combined_suggestions = list(set(suggest_texts + match_texts))

    return {"suggestions": combined_suggestions}

def expand_query_terms(query):
    terms = query.split()
    expanded_terms = set(terms)
    for term in terms:
        for syn in wn.synsets(term):
            for lemma in syn.lemmas():
                expanded_terms.add(lemma.name().replace('_', ' '))
    return list(expanded_terms)
