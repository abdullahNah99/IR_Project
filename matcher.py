from sklearn.metrics.pairwise import cosine_similarity
from index_creator import custom_preprocessor


def search(vectorizer, tfidf_matrix, dataset, query, top_n=10):   
    query = custom_preprocessor(query) 
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]    
    ids_column_name = dataset.columns[0]
    content_column_name = dataset.columns[1]
    top_ids = []
    top_docs = []
    for i in top_indices:
        top_ids.append(dataset.at[i, ids_column_name])    
        top_docs.append(dataset.at[i, content_column_name])    
    return top_ids, top_docs