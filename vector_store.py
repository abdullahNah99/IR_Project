import pinecone
import numpy as np



def upload_vectors_in_batches(index, tfidf_matrix, batch_size=100):
    num_rows = tfidf_matrix.shape[0]
    for start in range(0, num_rows, batch_size):
        end = min(start + batch_size, num_rows)
        batch_vectors = tfidf_matrix[start:end].toarray()
        batch_ids = [str(i) for i in range(start, end)]

        non_zero_indices = [i for i, vector in enumerate(batch_vectors) if np.any(vector)]
        filtered_batch_ids = [batch_ids[i] for i in non_zero_indices]
        filtered_batch_vectors = [batch_vectors[i].tolist() for i in non_zero_indices]
        
        vectors = list(zip(filtered_batch_ids, filtered_batch_vectors))
        if vectors:
            index.upsert(vectors)        

def storing(api_key, tfidf_matrix, index_name):    
    pinecone_client = pinecone.Pinecone(api_key=api_key)
    dimension = tfidf_matrix.shape[1]    
    if index_name not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(name=index_name, dimension=dimension, metric="cosine", spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1"))
    index = pinecone_client.Index(index_name)
    upload_vectors_in_batches(index, tfidf_matrix, batch_size=25)


def pinecone_search(vectorizer, dataset, api_key, index_name, query):
    pinecone_client = pinecone.Pinecone(api_key=api_key)    
    index = pinecone_client.Index(index_name)
    query_vactor = vectorizer.transform([query]).todense().tolist()[0]
    results = index.query(vector=query_vactor, top_k=10)    
    results = [int(match['id']) for match in results['matches']]
    ids_column_name = dataset.columns[0]
    content_column_name = dataset.columns[1]
    top_ids = []
    top_docs = []
    for i in results:
        top_ids.append(dataset.at[i, ids_column_name])    
        top_docs.append(dataset.at[i, content_column_name])    
    return top_ids, top_docs
