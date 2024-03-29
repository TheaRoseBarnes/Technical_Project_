# import modules
import pandas as pd
import numpy as np
import dtaidistance
import itertools

# The script DTW computes the independent DTW between all vital measurements for all patients, resulting in 
# an nxn distance matrix where n is the number of unique patient stays, and the matrix contains the values 
# quantifying the pairwise average optimal DTW alignment between patient vital measurement trajectories.

# For this script, the scaled_df outputted in Feature_Transformation.py is required.


def fill_mean(df, lst_features):
    for feature in lst_features:
        df[feature] = df[feature].fillna(df.groupby('stay_id')[feature].transform('mean'))
    return df

def fill_mode(df, lst_features):
    for feature in lst_features:
        df.loc[df[feature].isnull(), feature] = df['stay_id'].map(
            fast_mode(df, ['stay_id'], feature).set_index('stay_id')[feature])
    return df


def fast_mode(df, key_cols, value_col):
    return (df.groupby(key_cols + [value_col]).size()
              .to_frame('counts').reset_index()
              .sort_values('counts', ascending=False)
              .drop_duplicates(subset=key_cols)).drop(columns='counts')


# dependent DTW is commented out since it was not applied

# def depen_dtw(df, patients):
#     dictionary = {}
#     for stay in patients:
#         dictionary["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[
#             ['charttime', 'temperature', 'respiration', 'heart_rate', 'sats', 'systolic_bp',
#              'gcs_eye', 'gcs_verbal', 'gcs_motor']].sort_values(by='charttime').drop('charttime',
#                                                                                                     axis=1).to_numpy()
#     variable_names = list(dictionary.keys())
#     combs = list(itertools.combinations(variable_names, 2))
#     n = len(patients)
#     # df_stay_id = pd.DataFrame.from_records(variable_names, columns=['stay_id','results'])
#     stay = pd.DataFrame([dictionary]).melt()
#     dist = lambda a, b: dtw_ndim.distance(a, b)
#     distance_vectors = [dist(dictionary[pair[0]], dictionary[pair[1]]) for pair in combs]
#     # DTW_distance_matrix = squareform(np.array(distance_vectors))
#     DTW_distance_matrix = pd.DataFrame(squareform(np.array(distance_vectors)), columns=stay.variable.unique(),
#                                        index=stay.variable.unique())
#     #DTW_distance_matrix.to_csv('out_dtw_distance_matrix', index=False)
#     return DTW_distance_matrix


def indep_dtw(df, patients):
    dictionary = {}
    for stay in patients:
        dictionary["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[
            ['charttime', 'temperature', 'respiration', 'heart_rate', 'sats', 'systolic_bp', 'gcs_eye', 'gcs_verbal',
             'gcs_motor']].sort_values(by='charttime').drop('charttime', axis=1).to_numpy()
        variable_names = list(dictionary.keys())
        combs = list(itertools.combinations(variable_names, 2))
        n = len(patients)
        temperature = {}
        respiration = {}
        heart_rate = {}
        sats = {}
        systolic_bp = {}
        eye_gcs = {}
        verbal_gcs = {}
        motor_gcs = {}
    for stay in unique_patient_stays:
        temperature["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[
            ['charttime', 'temperature']].sort_values(by='charttime').drop('charttime', axis=1).to_numpy()
        respiration["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[
            ['charttime', 'respiration']].sort_values(by='charttime').drop('charttime', axis=1).to_numpy()
        heart_rate["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[
            ['charttime', 'heart_rate']].sort_values(by='charttime').drop('charttime', axis=1).to_numpy()
        sats["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[['charttime', 'sats']].sort_values(
            by='charttime').drop('charttime', axis=1).to_numpy()
        systolic_bp["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[
            ['charttime', 'systolic_bp']].sort_values(by='charttime').drop('charttime', axis=1).to_numpy()
        eye_gcs["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[['charttime', 'gcs_eye']].sort_values(
            by='charttime').drop('charttime', axis=1).to_numpy()
        verbal_gcs["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[
            ['charttime', 'gcs_verbal']].sort_values(by='charttime').drop('charttime', axis=1).to_numpy()
        motor_gcs["stay{0}".format(stay)] = (df.loc[df['stay_id'] == stay])[['charttime', 'gcs_motor']].sort_values(
            by='charttime').drop('charttime', axis=1).to_numpy()

    vitals = [temperature, respiration, heart_rate, sats, systolic_bp, eye_gcs, verbal_gcs, motor_gcs]
    t = list(temperature.values())
    DTW_t = dtaidistance.dtw.distance_matrix(t)
    r = list(respiration.values())
    DTW_r = dtaidistance.dtw.distance_matrix(r)
    hr = list(heart_rate.values())
    DTW_hr = dtaidistance.dtw.distance_matrix(hr)
    s = list(sats.values())
    DTW_s = dtaidistance.dtw.distance_matrix(s)
    sb = list(systolic_bp.values())
    DTW_sb = dtaidistance.dtw.distance_matrix(sb)
    eg = list(eye_gcs.values())
    DTW_eg = dtaidistance.dtw.distance_matrix(eg)
    vg = list(verbal_gcs.values())
    DTW_vg = dtaidistance.dtw.distance_matrix(vg)
    mg = list(motor_gcs.values())
    DTW_mg = dtaidistance.dtw.distance_matrix(mg)
    total_DTW_matrix = DTW_t + DTW_r + DTW_hr + DTW_s + DTW_sb + DTW_eg + DTW_vg + DTW_mg

    return total_DTW_matrix




if __name__ == '__main__':
    # Import the scaled_df, outputted from Feature_transformation.py
    df = pd.read_csv('')

    # Replace missing continuous vitals with the mean value across admission for each patient
    vitals = ['temperature', 'sats','systolic_bp','respiration','heart_rate']
    df = fill_mean(df, vitals)

    # Replace discrete vitals with the mode value across admission for each patient
    gcs_vitals = ['gcs_eye','gcs_verbal','gcs_motor']
    df = fill_mode(df, gcs_vitals)

    df = df.dropna()
    df['charttime'] = pd.to_datetime(df['charttime'])
    df = df.sort_values(by='charttime')
    unique_patient_stays = df.stay_id.unique().tolist()

    # Independent DTW is used in the pipeline, meaning there is a DTW distance matrix for each vital measurement.
    # In my analysis, I add these matrices together to perform dimensionality reduction and clustering on, however
    # if clustering of individual vital distance matrices is required, alter the indep_dtw function to return these
    # individual distance matrices.
    average_dtw_matrix = indep_dtw(df, unique_patient_stays)





       



