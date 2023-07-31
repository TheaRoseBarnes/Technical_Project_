# import modules
import umap.umap_ as umap
import pandas as pd
import hdbscan
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.cluster import adjusted_rand_score


# dimensionality reduction and clustering
def dimension_clustering(distance_matrix, n_neighbours, min_cluster_size, min_samples):
    trans = umap.UMAP(n_components=2, n_neighbors=n_neighbours, min_dist=0, random_state=42, metric='precomputed').fit(distance_matrix)
    #umap_2d = umap.UMAP(n_components=2, n_neighbors=n_neighbours, min_dist=0, random_state=42, metric='precomputed')
    out = coordinates = trans.embedding_
    out = pd.DataFrame(out, columns=['Column_A', 'Column_B'])
    clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=0.29, alpha=1.3)
    clusterer.fit(out)
    labels = clusterer.labels_
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sc = ax4.scatter(out['Column_A'], out['Column_B'], c=labels, cmap='viridis_r', alpha=0.5)
    ax4.legend(*sc.legend_elements(), title='cluster labels')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return out, labels, trans


# internal validation
def validation(min_samples_list, scoring, coordinates):
    min_cluster_size = [70, 80]
    list_scores_min_cluster_size = []
    for num in min_cluster_size:
        list_scores = []
        list_num_cluster = []
        for NUM in min_samples_list:
            clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=num, min_samples=NUM,
                                        cluster_selection_epsilon=0.29, alpha=1.3)
            clusterer.fit(coordinates)
            labels = clusterer.labels_
            num_clusters = len(np.unique(labels)) - 1
            list_num_cluster.append(num_clusters)
            score = scoring(coordinates, labels)
            list_scores.append(score)
        list_scores_min_cluster_size.append(list_scores)
    fig, ax1 = plt.subplots()
    ax1.plot(min_samples_list, list_scores, color="black", marker="o")
    ax1.set_xlabel("min samples", fontsize=14)
    ax1.set_ylabel(f"{scoring}", color="black", fontsize=14)
    axa = ax1.twinx()
    axa.set_ylabel("number of clusters", color="blue", fontsize=14)
    axa.plot(min_samples_list, list_num_cluster, color="blue", marker="o")
    plt.show()
    return list_scores, list_num_cluster, min_samples_list

if __name__ =='__main__':
    
    # import DTW distance matrix from DTW.py (average_dtw_matrix)
    dtw_distance_matrix = pd.read_csv('', header=None)

    # import unique patient id's from DTW.py (unique_patient_stays_df)
    stay_id_labels = pd.read_csv('')


    # UMAP and clustering
    umap_2d_coords, labels, trans = dimension_clustering(dtw_distance_matrix, 52, 75, 45)

    final_output_df = pd.DataFrame(stay_id_labels['stay_id'])
    final_output_df['clusters'] = labels.tolist()
    # final_output_df['variable'] = final_output_df['0']
    # final_output_df.rename(columns={'variable':'stay_id'}, inplace=True)
    final_output_df = final_output_df.reset_index()
    new_df = pd.merge(final_output_df, umap_2d_coords, left_index=True, right_index=True)

    plt.scatter(new_df['Column_A'], new_df['Column_B'], alpha = .6, c = new_df['clusters'], cmap = 'Paired',marker='.')
    plt.show()

    
    # accuracy / cluster validation

    list_scores, list_num_cluster, min_samples_list = validation([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80], silhouette_score, umap_2d_coords)
 
