"""Evaluation metrics for embeddings."""
import csv
import os
import numpy as np
from scipy.stats import pearsonr


def cosine_distance(emb1, emb2):
    """Compute cosine distance: (1 - cosine_similarity) / 2."""
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
    return (1 - sim) / 2


def compute_test_metrics(embeddings_dict, test_dir="test/results"):
    """Compute evaluation metrics on test set."""
    metrics = {}

    # ClinVar annotation using matched_pairs_labeled.csv
    clinvar_file = os.path.join(os.path.dirname(__file__), "data/matched_pairs_labeled.csv")
    if os.path.exists(clinvar_file):
        benign_distances, pathologic_distances = [], []
        with open(clinvar_file, "r") as f:
            for row in csv.DictReader(f):
                a, b = row["sample_a"], row["sample_b"]
                if a in embeddings_dict and b in embeddings_dict:
                    cd = cosine_distance(embeddings_dict[a], embeddings_dict[b])
                    if int(row["label"]) == 1:
                        benign_distances.append(cd)
                    elif int(row["label"]) == -1:
                        pathologic_distances.append(cd)
        if benign_distances and pathologic_distances:
            mean_benign = np.mean(benign_distances)
            mean_pathologic = np.mean(pathologic_distances)
            metrics.update({
                "cdd": mean_pathologic - mean_benign,
                "cd": np.mean(benign_distances + pathologic_distances)
            })

    # Mutation annotation
    mut_file = os.path.join(test_dir, "match_mut.csv")
    if os.path.exists(mut_file):
        distances_list, mut_counts = [], []
        
        with open(mut_file, "r") as f:
            for row in csv.DictReader(f):
                if row["sample_a"] in embeddings_dict and row["sample_b"] in embeddings_dict:
                    distances_list.append(cosine_distance(embeddings_dict[row["sample_a"]], embeddings_dict[row["sample_b"]]))
                    mut_counts.append(int(row["distance"]))
        
        if len(distances_list) > 1:
            pcc, pvalue = pearsonr(np.array(distances_list), np.array(mut_counts)) #np.log2(np.array(mut_counts)) / 9.0)
            metrics.update({"pcc": pcc})

    metrics.update({"mean": (metrics["cd"] + metrics["cdd"] + metrics["pcc"])/3} )
    
    return metrics
