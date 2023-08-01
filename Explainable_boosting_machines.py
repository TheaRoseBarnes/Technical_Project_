# import modules
import pandas as pd
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from collections import Counter
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTENC
import matplotlib.pyplot as plt



# import non time-dependent data
# import admissions table from MIMIC-IV data found in hosp module
df_admissions = pd.read_csv('')

# import icustays table from MIMIC-IV data found in icu module
icu_stays = pd.read_csv('/Users/theabarnes/Documents/Masters/Technical Project/Pycharm/Main_data/icustays.csv')

#cardiac_df = pd.read_csv('/Users/theabarnes/Documents/cardiac_arrests_2.csv/Sheet 1-cardiac_arrests.csv')

# select relevant features
df_admissions1 = df_admissions[['hadm_id','hospital_expire_flag','admission_type','admission_location']]

def Most_Common (lst):
    return max(((item, lst.count(item)) for item in set(lst)), key=lambda a: a[1])[0]

def Second_Most_Common(lst):
    return max(((item, lst.count(item)) for item in set(lst)), key=lambda a: a[1])[0]


def fast_mode(df, key_cols, value_col):

    return (df.groupby(key_cols + [value_col]).size()
                .to_frame('counts').reset_index()
                .sort_values('counts', ascending=False)
                .drop_duplicates(subset=key_cols)).drop(columns='counts')




# preprocess the main data with outcomes
def preprocess(main_data, death, stays,labels ):
    main_data['stay_id'] = main_data['stay_id'].astype('int64')
    labels['stay_id'] = labels['stay_id'].astype('int64')
    main_data['hadm_id'] = main_data['hadm_id'].astype('int64')
    stays['subject_id_x'] = stays['subject_id'].astype('int64')
    death['hadm_id'] = death['hadm_id'].astype('int64')
    prediction_data = pd.merge(left=main_data, right=death, on=['hadm_id'], how='inner')
    prediction_data['icd_code_list'] = prediction_data['GROUP_CONCAT(C2)'].str.split(',', expand=False)
    prediction_data['top_icd_codes'] = prediction_data['icd_code_list'].apply(
        lambda x: [word for word, word_count in Counter(x).most_common(3)])
    prediction_data['number_icd_codes'] = prediction_data['icd_code_list'].apply(lambda x: len(x))
    # prediction_raw_data['third_icd_code'] = prediction_raw_data['icd_code_list'].apply(lambda x: Third_Most_Common(x))df2["teams"].to_list(), columns=['team1', 'team2'])
    split_df = pd.DataFrame(prediction_data['top_icd_codes'].tolist(),
                            columns=['top_icd_code', 'second_icd_code', 'third_icd_code'])
    prediction_data = pd.concat([prediction_data, split_df], axis=1)
    prediction_data = pd.merge(left=stays, right=prediction_data, on=['subject_id'], how='outer')
    prediction_data['stay_id'] = prediction_data['stay_id_x']  # .astype('int64')
    prediction_data_clusters = pd.merge(left=prediction_data, right=labels, on=['stay_id'], how='inner')
    prediction_data_clusters = prediction_data_clusters[pd.notnull(prediction_data_clusters['clusters'])]
    # prediction_data_clusters = prediction_data_clusters.replace(, np.nan)
    prediction_data_clusters = prediction_data_clusters[
        ['subject_id', 'stay_id_x', 'clusters', 'AVG(anchor_age)', 'MIN(gender)', 'AVG(temperature)',
         'AVG(respiration)', 'AVG(heart_rate)', 'AVG(sats)', 'AVG(systolic_bp)', 'MAX(gcs_motor)', 'MAX(gcs_verbal)',
         'MAX(gcs_eye)', 'top_icd_code', 'second_icd_code', 'number_icd_codes','los' ,'first_careunit','last_careunit','admission_type','admission_location',
          'hospital_expire_flag']]
    prediction_data_clusters["AVG(temperature)"] = prediction_data_clusters["AVG(temperature)"].fillna(
        prediction_data_clusters.groupby('stay_id_x')['AVG(temperature)'].transform('mean'))
    prediction_data_clusters["AVG(sats)"] = prediction_data_clusters["AVG(sats)"].fillna(
        prediction_data_clusters.groupby('stay_id_x')['AVG(sats)'].transform('mean'))
    prediction_data_clusters["AVG(systolic_bp)"] = prediction_data_clusters["AVG(systolic_bp)"].fillna(
        prediction_data_clusters.groupby('stay_id_x')['AVG(systolic_bp)'].transform('mean'))
    prediction_data_clusters["AVG(heart_rate)"] = prediction_data_clusters["AVG(heart_rate)"].fillna(
        prediction_data_clusters.groupby('stay_id_x')['AVG(heart_rate)'].transform('mean'))
    prediction_data_clusters["AVG(respiration)"] = prediction_data_clusters["AVG(respiration)"].fillna(
        prediction_data_clusters.groupby('stay_id_x')['AVG(respiration)'].transform('mean'))
    #prediction_data_clusters['los'] = prediction_data_clusters['los'].fillna(
        #prediction_data_clusters.groupby('stay_id_x')['los'].transform('mean'))
    prediction_data_clusters['number_icd_codes'] = prediction_data_clusters['number_icd_codes'].fillna(
        prediction_data_clusters.groupby('stay_id_x')['number_icd_codes'].transform('mean'))
    # prediction_data_clusters['MAX(gcs_eye)'] = prediction_data_clusters['MAX(gcs_eye)'].replace('nan', 'AAA')
    # unsure = fast_mode(prediction_data_clusters, ['stay_id_x'], 'MAX(gcs_eye)')
    prediction_data_clusters.loc[prediction_data_clusters['MAX(gcs_eye)'].isnull(), 'MAX(gcs_eye)'] = \
    prediction_data_clusters['stay_id_x'].map(
        fast_mode(prediction_data_clusters, ['stay_id_x'], 'MAX(gcs_eye)').set_index('stay_id_x')['MAX(gcs_eye)'])
    prediction_data_clusters.loc[prediction_data_clusters['MAX(gcs_verbal)'].isnull(), 'MAX(gcs_verbal)'] = \
    prediction_data_clusters['stay_id_x'].map(
        fast_mode(prediction_data_clusters, ['stay_id_x'], 'MAX(gcs_verbal)').set_index('stay_id_x')['MAX(gcs_verbal)'])
    prediction_data_clusters.loc[prediction_data_clusters['MAX(gcs_motor)'].isnull(), 'MAX(gcs_motor)'] = \
    prediction_data_clusters['stay_id_x'].map(
        fast_mode(prediction_data_clusters, ['stay_id_x'], 'MAX(gcs_motor)').set_index('stay_id_x')['MAX(gcs_motor)'])
    prediction_data_clusters.loc[prediction_data_clusters['top_icd_code'].isnull(), 'top_icd_code'] = \
    prediction_data_clusters['stay_id_x'].map(
        fast_mode(prediction_data_clusters, ['stay_id_x'], 'top_icd_code').set_index('stay_id_x')['top_icd_code'])
    # prediction_data_clusters['MAX(gcs_eye)'] = prediction_data_clusters.groupby('stay_id_x')['MAX(gcs_eye)'].apply(lambda x: x.fillna(x.mode()[0]))
    # prediction_data_clusters['MAX(gcs_verbal)'] = prediction_data_clusters.groupby('stay_id_x')['MAX(gcs_verbal)'].apply(lambda x: x.fillna(x.mode()[0]))
    # prediction_data_clusters['MAX(gcs_motor)'] = prediction_data_clusters.groupby('stay_id_x')['MAX(gcs_motor)'].apply(lambda x: x.fillna(x.mode()[0]))
    prediction_data_clusters['first_careunit'] = prediction_data_clusters.groupby('stay_id_x')['first_careunit'].apply(
        lambda x: x.fillna(x.mode()[0]))
    #prediction_data_clusters['last_careunit'] = prediction_data_clusters.groupby('stay_id_x')['last_careunit'].apply(
        #lambda x: x.fillna(x.mode()[0]))
    # prediction_data_clusters['top_icd_code'] = prediction_data_clusters.groupby('stay_id_x')['top_icd_code'].apply(lambda x: x.fillna(x.mode()[0]))
    prediction_data_clusters = prediction_data_clusters.dropna(axis=0)
    prediction_data_clusters['hospital_expire_flag'] = prediction_data_clusters['hospital_expire_flag'].astype(int)
    num_clusters = prediction_data_clusters.clusters.unique()
    return prediction_data_clusters, num_clusters

# get individual dataframes for each cluster
def cluster_data(df, Folds = 10):
    train_cols = df.columns[3:-1]
    label = df.columns[-1]
    X = df[train_cols]
    y = df[label]
    seed = 50
    groups = df['stay_id_x']
    k_fold = StratifiedGroupKFold(n_splits= Folds)
    predictor = ExplainableBoostingClassifier(random_state=seed)
    cat_col_index = [1, 7, 8, 9, 10, 11, 13, 14, 15]
    over = SMOTENC(categorical_features=cat_col_index, random_state=50, sampling_strategy=0.9)

    y_real = []
    y_proba = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    sens_scores = []
    features = pd.DataFrame(columns=['type', 'names', 'scores'])

    for i, (train_index, test_index) in enumerate(k_fold.split(X, y, groups=groups)):
        Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
        ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
        Xtrain, ytrain = over.fit_resample(Xtrain, ytrain)
        predictor.fit(Xtrain, ytrain)
        pred_proba = predictor.predict_proba(Xtest)
        y_pred = predictor.predict(Xtest)
        precision, recall, _ = precision_recall_curve(ytest, pred_proba[:, 1])
        lab = 'Fold %d AUC=%.4f' % (i + 1, auc(recall, precision))
        y_real.append(ytest)
        y_proba.append(pred_proba[:, 1])

        # feature importances
        predictor_global = predictor.explain_global()
        feat = predictor_global.data(key=None)
        features_new = pd.DataFrame(feat)
        features = pd.concat([features, features_new])

        # calculating performance metrics
        precision = precision_score(ytest, y_pred)
        recall = recall_score(ytest, y_pred)
        sens = recall_score(ytest, y_pred, pos_label=0)
        f1 = f1_score(ytest, y_pred, average='macro')
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        sens_scores.append(sens)

    features = features[['names', 'scores']]
    new_features = (features.groupby(['names'])['scores'].agg([('scores', 'mean'), ('s_t_v', 'std')])).reset_index()
    new_features.sort_values(by='scores', ascending=False)
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    # yhat_ = numpy.concatenate(yhat)
    # lr_auc_ = numpy.concatenate(lr_auc)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    # lr_f1, lr_auc = f1_score(y_test, yhat_), auc(lr_recall, lr_precision)
    lr_fpr, lr_tpr, _ = roc_curve(y_real, y_proba)

    return sens_scores, f1_scores, recall_scores, precision_scores,lab, precision, recall, lr_fpr, lr_tpr, new_features



if __name__ =='__main__':

    # import prediction data and cluster labels
    # import coordinates outputted from clustering.py file
    coordinates =pd.read_csv('', na_values=np.nan)

    # import prediction_data table
    main_data_24 = pd.read_csv('/Users/theabarnes/Documents/Masters/Technical Project/Pycharm/24_prediction_2.csv', na_values=np.nan)
    

    # # CARDIAC ARREST ADDITON
    # cardiac_data = pd.read_csv('/Users/theabarnes/Documents/Masters/Technical Project/Pycharm/cardic_24_prediction_data.csv', na_values=np.nan)
    # cardiac_labels = cardiac_data.drop_duplicates('stay_id')
    # cardiac_labels = cardiac_labels['stay_id']

    labels = coordinates[['stay_id', 'clusters']]
    data, num = preprocess(main_data_24, df_admissions1, icu_stays,labels )


    # # ANOTHER CARDIAC ARREST ADDITION
    # c_data, num_clusters = preprocess(cardiac_data, df_admissions, icu_stays, labels)
    # data_minus = data[~data.stay_id_x.isin(cardiac_labels)]
    # data_minus['cardiac_flag'] = 0
    # c_data['cardiac_flag'] = 1
    # combined = c_data.append(data_minus)
    # combined = combined.drop(['hospital_expire_flag'], axis=1)
    # data = combined

    df_0 = data.loc[data['clusters'] == 0]
    df_1 = data.loc[data['clusters'] == 1]
    df_2 = data.loc[data['clusters'] == 2]
    df_3 = data.loc[data['clusters'] == 3]
    df_4 = data.loc[data['clusters'] == 4]
    df_5 = data.loc[data['clusters'] == 5]


    sens_scores0, f1_scores0, recall_scores0, precision_scores0,lab0, precision0, recall0, lr_fpr0, lr_tpr0, new_features0 = cluster_data(df_0, Folds=10)
    sens_scores1, f1_scores1, recall_scores1, precision_scores1,lab1, precision1, recall1 ,lr_fpr1, lr_tpr1, new_features1 = cluster_data(df_1, Folds=10)
    sens_scores2, f1_scores2, recall_scores2, precision_scores2,lab2, precision2, recall2, lr_fpr2, lr_tpr2, new_features2 =cluster_data(df_2, Folds=10)
    sens_scores3, f1_scores3, recall_scores3, precision_scores3,lab3, precision3, recall3, lr_fpr3, lr_tpr3, new_features3=cluster_data(df_3, Folds=10)
    sens_scores4, f1_scores4, recall_scores4, precision_scores4,lab4, precision4, recall4, lr_fpr4, lr_tpr4, new_features4 = cluster_data(df_4, Folds=10)
    sens_scores5, f1_scores5, recall_scores5, precision_scores5,lab5, precision5, recall5, lr_fpr5, lr_tpr5, new_features5 = cluster_data(df_5, Folds=10)

    sens_scores_all = sens_scores1 + sens_scores2 + sens_scores3 + sens_scores4 + sens_scores5
    sens_avg = np.mean(sens_scores_all)

    f1_scores_all = f1_scores1 + f1_scores2 + f1_scores3 + f1_scores4 + f1_scores5
    f1_avg = np.mean(f1_scores_all)

    prec_scores_all = precision_scores1   + precision_scores2  + precision_scores3 + precision_scores4 + precision_scores5
    prec_avg = np.mean(prec_scores_all)

    recall_scores_all = recall_scores1  + recall_scores2 + recall_scores3 + recall_scores4 + recall_scores5
    recall_avg = np.mean(recall_scores_all)




    # PLOTS -
    #fig = plt.figure()
    #ax = fig.add_axes([1, 1, 1, 1])
    font = {'family': 'normal', 'weight': 'normal'}
    plt.rc('font', **font)
    plt.plot(lr_fpr0, lr_tpr0, label='Cluster 0: AUC=%.3f' % (auc(recall0, precision0)))
    plt.plot(lr_fpr1, lr_tpr1, label='Cluster 1: AUC=%.3f' % (auc(recall1, precision1)))
    plt.plot(lr_fpr2, lr_tpr2, label='Cluster 2: AUC=%.3f' % (auc(recall2, precision2)))
    plt.plot(lr_fpr3, lr_tpr3, label='Cluster 3: AUC=%.3f' % (auc(recall3, precision3)))
   # plt.plot(lr_fpr4, lr_tpr4, label='Cluster 4: AUC=%.3f' % (auc(recall4, precision4)))
   # plt.plot(lr_fpr5, lr_tpr5, label='Cluster 5: AUC=%.3f' % (auc(recall5, precision5)))
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    plt.show()
    # #plt.savefig('False_positive_mortaility.png', bbox_inches='tight')

    #fig = plt.figure()
    #ax = fig.add_axes([1, 1, 1, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    font = {'family': 'normal', 'weight': 'normal'}
    plt.rc('font', **font)
    plt.plot(recall0, precision0, label='Cluster 0')
    plt.plot(recall1, precision1, label='Cluster 1')
    plt.plot(recall2, precision2, label='Cluster 2')
    plt.plot(recall3, precision3, label='Cluster 3')
    #plt.plot(recall4, precision4, label='Cluster 4')
    #plt.plot(recall5, precision5, label='Cluster 5')
    plt.legend(loc='lower left')
    plt.show()
