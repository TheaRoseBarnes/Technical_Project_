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
    # import DTW distance matrices
    dtw_distance_matrix = pd.read_csv('/Users/theabarnes/Documents/Masters/6000_inpend_dtw_dm.csv', header=None)
    dtw_distance_matrix = dtw_distance_matrix.drop([196, 424, 248, 5615, 3312])
    dtw_distance_matrix = dtw_distance_matrix.drop(columns=dtw_distance_matrix.columns[[248, 196, 424, 5615, 3312]])

    # import 4 hours
    dm_4 = pd.read_csv('/Users/theabarnes/Documents/Masters/Technical Project/average_dtw_matrix_4hours.csv', header=None)
    dm_4 = dm_4.drop([196, 424, 248, 5615, 3312])
    dm_4 = dm_4.drop(columns=dm_4.columns[[248, 196, 424, 5615, 3312]])


    # import 72 hours
    dm_72 = pd.read_csv('/Users/theabarnes/Documents/Masters/average_DTW_matrix_72_hours.csv', header=None)
    dm_72 = dm_72.drop([196, 424, 248, 5615, 3312])
    dm_72 = dm_72.drop(columns=dm_72.columns[[248, 196, 424, 5615, 3312]])

    # import 24 hours
    dm_24 = pd.read_csv('/Users/theabarnes/Documents/Masters/average_DTW_matrix_24.csv', header=None)
    dm_24 = dm_24.drop([196, 424, 248, 5615, 3312])
    dm_24 = dm_24.drop(columns=dm_24.columns[[248, 196, 424, 5615, 3312]])

    # import 1 week
    dm_1week = pd.read_csv('/Users/theabarnes/Documents/Masters/average_DTW_matrix_1week.csv', header=None)
    dm_1week = dm_1week.drop([196, 424, 248, 5615, 3312])
    dm_1week = dm_1week.drop(columns=dm_1week.columns[[248, 196, 424, 5615, 3312]])



    stay_id_labels = pd.read_csv('/Users/theabarnes/Documents/Masters/indep_6000_cluster_labels.csv')
    #final_output_df = pd.DataFrame(stay_id_labels['0'])
    final_output_df = stay_id_labels.drop([196, 424, 248, 5615, 3312])

    # UMAP and clustering
    umap_2d_coords, labels2, trans = dimension_clustering(dm_24, 52, 75, 45)
    umap_2d_coords3, labels2, trans3 = dimension_clustering(dm_72, 52, 75, 30)
    umap_2d_coords4, labels2, trans4 = dimension_clustering(dm_1week, 52, 75, 50)
    umap_2d_coords5, labels2, trans5 = dimension_clustering(dm_4, 52, 75, 30)
    #


    # final_output_df['clusters'] = labels2.tolist()
    # final_output_df['variable'] = final_output_df['0']
    # final_output_df.rename(columns={'variable':'stay_id'}, inplace=True)
    # final_output_df = final_output_df.reset_index()
    # new_df = pd.merge(final_output_df, umap_2d_coords5, left_index=True, right_index=True)

    # plt.scatter(new_df['Column_A'], new_df['Column_B'], alpha = .6, c = new_df['clusters'], cmap = 'Paired',marker='.')
    # plt.show()
    # accuracy / cluster validation

    list_scores, list_num_cluster, min_samples_list = validation([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80], silhouette_score, umap_2d_coords5)
    list_scores, list_num_cluster, min_samples_list = validation([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80], davies_bouldin_score, umap_2d_coords5)
    labels = final_output_df['clusters']
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18,4))
    labels = final_output_df['clusters']
    test_embedding2 = trans5.transform(dm_4)
    plt.rcParams.update({'font.size': 17})
    ax1.scatter(test_embedding2[:, 0], test_embedding2[:, 1], s=5, c=labels, cmap='Set1_r')
    #plt.show()
    clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=52, min_samples=52, cluster_selection_epsilon=0.29,alpha=1.3)
    clusterer.fit(test_embedding2)
    labels_24 = clusterer.labels_
    #plt.scatter(test_embedding2[:, 0], test_embedding2[:, 1], s=5, c=labels_24, cmap='viridis_r')
    #plt.show()
    score_24 = adjusted_rand_score(labels, labels_24)
    #ax1.text(-5, 60, f'Rand Index: {score_24}', fontsize=17)

    #test_embedding = trans.transform(dm_48)
    #test_embedding, labels2, trans = dimension_clustering(dm_48, 52, 75, 30)
    test_embedding = trans.transform(dm_24)
    ax2.scatter(test_embedding[:, 0], test_embedding[:, 1], s=5, c=labels, cmap='Set1_r')
    #plt.show()
    clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=52, min_samples=52,cluster_selection_epsilon=0.29, alpha=1.3)
    clusterer.fit(test_embedding)
    labels_48 = clusterer.labels_
    #plt.scatter(test_embedding[:, 0], test_embedding[:, 1], s=5, c=labels_48, cmap='viridis_r')
    #plt.show()
    score_48 = adjusted_rand_score(labels, labels_48)
    #ax2.text(-5, 60, f'Rand Index: {score_48}', fontsize=17)

    #test_embedding3 = trans.transform(dm_72)
    #test_embedding3, labels3, trans = dimension_clustering(dm_72, 52, 75, 30)
    test_embedding3 = trans3.transform(dm_72)
    ax3.scatter(test_embedding3[:, 0], test_embedding3[:, 1], s=5, c=labels, cmap='Set1_r')
    #plt.show()
    clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=52, min_samples=52, cluster_selection_epsilon=0.29,alpha=1.3)
    clusterer.fit(test_embedding3)
    labels_72 = clusterer.labels_
    #plt.scatter(test_embedding3[:, 0], test_embedding3[:, 1], s=5, c=labels_72, cmap='viridis_r')
    #plt.show()
    score_72 = adjusted_rand_score(labels, labels_72)
    #ax3.text(-5, 60, f'Rand Index: {score_72}', fontsize=17)

    #test_embedding4 = trans.transform(dm_1week)
    #test_embedding4, labels4, trans = dimension_clustering(dm_1week, 52, 75, 30)
    test_embedding4 = trans4.transform(dm_1week)
    sc =ax4.scatter(test_embedding4[:, 0], test_embedding4[:, 1], s=5, c=labels, cmap='Set1_r')
    clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=52, min_samples=52, cluster_selection_epsilon=0.29,alpha=1.3)
    clusterer.fit(test_embedding4)
    labels_1week = clusterer.labels_
    #plt.scatter(test_embedding4[:, 0], test_embedding4[:, 1], s=5, c=labels_1week, cmap='viridis_r')
    #plt.show()
    score_1week = adjusted_rand_score(labels, labels_1week)
    #ax4.text(-5, 60, f'Rand Index: {score_1week}', fontsize=17)
    font = {'size': 17}
    plt.rcParams.update({'font.size': 17})
    plt.rc('xtick', labelsize=17)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=17)
    plt.rc('font', **font)
    ax1.title.set_text('First 4 hours')
    ax2.title.set_text('First 24 hours')
    ax3.title.set_text('First 72 hours')
    ax4.title.set_text('1 week')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    ax3.set_xlabel('Dimension 1')
    ax3.set_ylabel('Dimension 2')
    ax4.set_xlabel('Dimension 1')
    ax4.set_ylabel('Dimension 2')
    fig.tight_layout(pad=0.2)
    #ax1.legend(*sc.legend_elements(),ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.3),title='entire admission cluster labels')
    ax1.legend(*sc.legend_elements(),loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=7,title='entire admission cluster labels')
    fig.subplots_adjust(bottom=0.2)
    plt.show()

    #test_embedding5 = trans.transform(dm_1week)
    # #test_embedding4, labels4, trans = dimension_clustering(dm_1week, 52, 75, 30)
    test_embedding5 = trans5.transform(dm_4)
   # sc =ax4.scatter(test_embedding4[:, 0], test_embedding4[:, 1], s=5, c=labels, cmap='viridis_r')
    #clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=52, min_samples=52, cluster_selection_epsilon=0.29,alpha=1.3)
    # clusterer.fit(test_embedding4)
    # labels_1week = clusterer.labels_
    plt.scatter(test_embedding5[:, 0], test_embedding5[:, 1], s=5, c=labels, cmap='Set1')
    plt.show()
