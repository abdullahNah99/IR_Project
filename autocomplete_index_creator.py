from elasticsearch import Elasticsearch
import csv
from datetime import datetime

es = Elasticsearch(
    ['http://127.0.0.1:9200'],
    request_timeout=300  
)

# إنشاء فهرس الإكمال التلقائي إذا لم يكن موجودًا
if es.indices.exists(index='autocomplete'):
    es.indices.delete(index='autocomplete')

es.indices.create(
    index='autocomplete',
    body={
        'settings': {
            'analysis': {
                'analyzer': {
                    'autocomplete': {
                        'type': 'custom',
                        'tokenizer': 'standard',
                        'filter': ['lowercase', 'autocomplete_filter']
                    }
                },
                'filter': {
                    'autocomplete_filter': {
                        'type': 'edge_ngram',
                        'min_gram': 2,
                        'max_gram': 20
                    }
                }
            }
        },
        'mappings': {
            'properties': {
                'query': {
                    'type': 'completion',
                    'analyzer': 'autocomplete',
                    'search_analyzer': 'standard'
                },
                'query_text': {  # إضافة حقل نصي جديد
                    'type': 'text'
                },
                'timestamp': {
                    'type': 'date'
                },
                'success': {
                    'type': 'boolean'
                }
            }
        }
    }
)

# قراءة البيانات من ملف CSV وإضافتها إلى Elasticsearch
with open('queries2.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        es.index(index='autocomplete', body={'query': row['query'], 'query_text': row['query'], 'timestamp': datetime.now(), 'success': True})