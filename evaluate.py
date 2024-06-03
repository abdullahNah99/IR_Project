import csv
import requests


def search(query, dataset):
    data = {"query": query, "dataset": dataset}
    response = requests.post("http://127.0.0.1:8001/matching", json=data)
    return response.json().get("top_ids")


def get_relevant_id_from_qrel(min_rel_val, query_id, csv_file):
    relevant_ids = []
    relevance_scores = {}
    # Read data from CSV file
    with open(csv_file, "r") as file:
        reader = csv.reader(file)

        for row in reader:
            if row[0] == query_id and int(row[3]) >= min_rel_val:
                doc_id = row[2]
                relevance = int(row[3])
                if relevance >= min_rel_val:
                    relevance = 1
                else:
                    relevance = 0
                relevant_ids.append(doc_id)
                relevance_scores[doc_id] = relevance
    return relevant_ids, relevance_scores


def precision_at_k(retrieved_docs, relevant_docs, relevance_scores, k):
    relevant_in_top_k = sum(1 for doc in retrieved_docs[:k] if doc in relevant_docs)
    precision = relevant_in_top_k / k
    return precision


def calculate_recall(min_rel_val, retrieved_docs, relevant_docs, relevance_scores):
    relevant_retrieved = sum(
        1
        for doc in retrieved_docs
        if doc in relevant_docs and relevance_scores[doc] > 0
    )
    total_relevant = sum(1 for rel in relevance_scores.values() if rel > 0)
    if total_relevant == 0:
        return 0
    recall = relevant_retrieved / total_relevant
    return recall


def average_precision_at_k(retrieved_docs, relevant_docs, relevance_scores, k):
    precision_sum = 0.0
    relevant_count = 0

    for i in range(min(k, len(retrieved_docs))):
        if retrieved_docs[i] in relevant_docs:
            relevant_count += 1
            precision_at_i = precision_at_k(
                retrieved_docs, relevant_docs, relevance_scores, i + 1
            )
            precision_sum += (
                precision_at_i * relevance_scores[retrieved_docs[i]]
            )  # Multiply by relevance score

    if relevant_count == 0:
        return 0

    average_precision = precision_sum / relevant_count
    return average_precision


def reciprocal_rank_at_k(
    min_rel_val, retrieved_docs, relevant_docs, relevance_scores, k=10
):
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs and relevance_scores[doc] >= min_rel_val:
            return 1 / (i + 1)
    return 0


def process_queries(dataset, min_rel_val, queries_file, qrels_file, k=10):
    precision_results = []
    recall_results = []
    queries_results = []
    mrr_results = []
    ap_values = []

    with open(queries_file, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            try:
                query_id = row[0]
                query = row[1]

                retrieved_docs = search(
                    query, dataset
                )  # Assuming search function is defined elsewhere
                relevant_docs, relevance_scores = get_relevant_id_from_qrel(
                    min_rel_val, query_id, qrels_file
                )

                precision = precision_at_k(
                    retrieved_docs, relevant_docs, relevance_scores, k
                )
                precision_percentage = precision * 100
                precision_results.append((query_id, precision_percentage))

                recall = calculate_recall(
                    min_rel_val, retrieved_docs, relevant_docs, relevance_scores
                )
                recall_percentage = recall * 100
                recall_results.append((query_id, recall_percentage))

                ap = average_precision_at_k(
                    retrieved_docs, relevant_docs, relevance_scores, k
                )
                ap_values.append(ap)

                rr = reciprocal_rank_at_k(
                    min_rel_val, retrieved_docs, relevant_docs, relevance_scores, k
                )
                mrr_results.append(rr)

                print(
                    f"Query ID: {query_id}, Precision@{k}: {precision_percentage:.2f}%, Recall@{k}: {recall_percentage:.2f}%, AP@{k}: {ap*100:.2f}"
                )

            except Exception as e:
                print(f"Error processing query: {query_id}. Error: {e}")

    final_map = sum(ap_values) / len(ap_values) * 100
    final_mrr = sum(mrr_results) / len(mrr_results) * 100

    return precision_results, recall_results, final_map, final_mrr
