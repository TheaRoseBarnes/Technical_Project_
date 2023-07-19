# This script performs data transformation on vital measurement features based on their distribution characteristics.

# Used tables: vitals_grouby_30min

# import packages
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import scale


# import data
df = pd.read_csv('')
df = df[['stay_id','charttime','subject_id','hadm_id', 'AVG(temperature)','AVG(respiration)','AVG(heart_rate)','AVG(sats)','AVG(systolic_bp)','MAX(gcs_verbal)','MAX(gcs_motor)','MAX(gcs_eye)']]

# fliter range values
# temp - 30 – 42 C which is 86- 107.6
new_df = df[df['AVG(temperature)'].between(86, 107.6)]
# heart rate 40-300
new_df = new_df[new_df['AVG(heart_rate)'].between(40,300)]
# respirtaiory 6-50
new_df = new_df[new_df['AVG(respiration)'].between(6,50)]
# blood pressure 40-200
new_df = new_df[new_df['AVG(systolic_bp)'].between(40,200)]
# SATS 0-160
new_df = new_df[new_df['AVG(sats)'].between(0,160)]



# glasco coma scale values
di = {'Confused': 4, 'Oriented': 5, 'No Response-ETT': 1,'No Response': 1, 'Incomprehensible sounds': 2, 'Inappropriate Words': 3,'Obeys Commands': 6, 'Flex-withdraws': 4, 'Localizes Pain': 5, 'No response': 1, 'Abnormal Flexion': 3,'To Speech':3,'Spontaneously':4,'To Pain':2,'None':1, 'Abnormal extension': 2}

new_df['MAX(gcs_eye)'].replace(di, inplace=True)
new_df['MAX(gcs_verbal)'].replace(di, inplace=True)
new_df['MAX(gcs_motor)'].replace(di, inplace=True)

new_all_vitals_main = new_df

print(new_df.describe())
new_df.info()
new_df.isnull().sum()


# look at each vitals distribution
'''
# TEMP is normal
temp = new_df[new_df['temperature'].notna()]
temp['temperature'].plot(kind='hist', edgecolor='black')
plt.show()

# not normal
heart_rate = new_df[new_df['heart_rate'].notna()]
heart_rate['heart_rate'].plot(kind='hist', edgecolor='black')
plt.show()

# normal
systolic_bp = new_df[new_df['systolic_bp'].notna()]
systolic_bp['systolic_bp'].plot(kind='hist', edgecolor='black')
plt.show()

# normal
respiration = new_df[new_df['respiration'].notna()]
respiration['respiration'].plot(kind='hist', edgecolor='black')
plt.show()

# not normal
sats = new_df[new_df['sats'].notna()]
sats['sats'].plot(kind='hist', edgecolor='black')
plt.show()


# normal
eye = new_df[new_df['gcs_eye'].notna()]
eye['gcs_eye'].plot(kind='hist', edgecolor='black')
plt.show()


# not normal
motor = new_df[new_df['gcs_motor'].notna()]
motor['gcs_motor'].plot(kind='hist', edgecolor='black')
plt.show()

# not normal
verbal = new_df[new_df['MIN(gcs_verbal)'].notna()]
verbal['MIN(gcs_verbal)'].plot(kind='hist', edgecolor='black')
plt.show()
'''
#
# def logit(data, min=-1, max=-1):
#     if min == -1 and max == -1:
#         print("Getting new min max")
#         min = data.min() - 1
#         max = data.max() + 1
#
#     dx = 1 / (max - min)
#     print(f"Min: {min} \t Max: {max} \t Range: {1 / dx}")
#     x = (data - min) * dx
#     print(f"xMin: {x.min()} \t xMax: {x.max()} \t xRange: {x.max() - x.min()}")
#     s = np.log(x / (1 - x))
#     print(f"sMin: {s.min()} \t sMax: {s.max()} \t sRange: {s.max() - s.min()}")
#     # return scale(s), (min, max)
#     return s, (min, max)

# scale to unit variance for normal distributed features, and min-max for skewed distributions
def scale_features(data, feature_list, dict_scaling):
    df_new = pd.DataFrame()
    df_new['stay_id'] = data['stay_id']
    df_new['charttime'] = data['charttime']
    for feature in feature_list:
        # df_new[feature] = dict_scaling[feature](data[feature])[0]
        if dict_scaling[feature] == "scale":
            df_new[feature] = scale(data[feature])
        elif dict_scaling[feature] == "not":
            df_new[feature] = \
                preprocessing.minmax_scale(data[feature], axis=0)
    return df_new

scaling = {'temperature':'scale','heart_rate':'not','systolic_bp':'scale','respiration':'scale','sats':'not','gcs_eye':'scale','gcs_motor':'not', 'gcs_verbal':'not'}
feature_list1 = ['temperature','heart_rate','systolic_bp', 'respiration','sats', 'gcs_eye', 'gcs_motor','gcs_verbal' ]

scaled_df = scale_features(new_df, feature_list1, scaling)
#stay = new_df[['stay_id','charttime']]
#scaled_df = pd.merge(scaled_df,stay, left_index=True, right_index=True)

