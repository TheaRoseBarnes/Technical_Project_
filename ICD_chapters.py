import pandas as pd
from icd9cms.icd9 import search
import simple_icd_10_cm as cm


df_icd = pd.read_csv('/Users/theabarnes/Documents/Masters/Technical Project/Pycharm/Technical_Project/d_icd_diagnoses.csv',skipinitialspace=True)
df_icd_9 = df_icd[df_icd['icd_version'] == 9]
df_icd_10 = df_icd[df_icd['icd_version'] == 10]


# For ICD_9 codes:
parent_icd_9 = []
final_parent_icd_9 = []

for code in df_icd_9['icd_code']:
    diagnosis = search(code)
    if diagnosis is None:
        final_parent_icd_9.append('Nan')
    else:
        first_parent = diagnosis.parent
        while first_parent is not None:
            first_parent = first_parent.parent
            parent_icd_9.append(first_parent)
        final_parent_icd_9.append(parent_icd_9[-2])
df_icd_9['parent_node'] = final_parent_icd_9


unique_9_chapters = df_icd_9['parent_node'].nunique()
unique_9_diagnosis = df_icd_9['icd_code'].nunique()



# For 10 ICD CODES
final_parent_icd_10 = []
parent_icd_10 = []

for value in df_icd_10['icd_code']:
#for value in huh:
    if cm.is_valid_item(value) is False:
        final_parent_icd_10.append('Nan')
    else:
        FirstParent = cm.get_parent(value)
        while cm.is_chapter_or_block(FirstParent) is False:
            FirstParent = cm.get_parent(FirstParent)
            parent_icd_10.append(FirstParent)
        final_parent_icd_10.append(parent_icd_10[-1])
df_icd_10['parent_node'] = final_parent_icd_10


chap_list = []
for block in df_icd_10['parent_node']:
    if block == 'Nan':
        chap_list.append('Nan')
    else:
        chap = cm.get_parent(block)
        if cm.is_chapter(chap)==True:
            desc = cm.get_description(chap)
            chap_list.append(desc)
        else:
            fuff = cm.get_ancestors(chap)
            desc_fuff = cm.get_description(fuff[-1])
            chap_list.append(desc_fuff)

df_icd_10['chapter'] = chap_list




vv = df_icd_10['chapter'].nunique()
yy = df_icd_10['icd_code'].nunique()


# Create a mapping which turns the icd_9 chapters into the same as the icd_10
Mapping = {'001-139:Infectious And Parasitic Diseases:None': 'Certain infectious and parasitic diseases (A00-B99)',
'140-239:Neoplasms:None' : 'Neoplasms (C00-D49)',
 '240-279:Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders:None' : 'Endocrine, nutritional and metabolic diseases (E00-E89)',
 '280-289:Diseases Of The Blood And Blood-Forming Organs:None' : 'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism (D50-D89)',
 '290-319:Mental Disorders:None': 'Mental, Behavioral and Neurodevelopmental disorders (F01-F99)',
 '320-389:Diseases Of The Nervous System And Sense Organs:None' : 'Diseases of the nervous system (G00-G99)',
 '390-459:Diseases Of The Circulatory System:None' : 'Diseases of the circulatory system (I00-I99)',
 '460-519:Diseases Of The Respiratory System:None' : 'Diseases of the respiratory system (J00-J99)',
 '520-579:Diseases Of The Digestive System:None': 'Diseases of the digestive system (K00-K95)',
 '580-629:Diseases Of The Genitourinary System:None': 'Diseases of the genitourinary system (N00-N99)',
 '630-679:Complications Of Pregnancy, Childbirth, And The Puerperium:None': 'Pregnancy, childbirth and the puerperium (O00-O9A)',
 '680-709:Diseases Of The Skin And Subcutaneous Tissue:None':'Diseases of the skin and subcutaneous tissue (L00-L99)',
 '710-739:Diseases Of The Musculoskeletal System And Connective Tissue:None':'Diseases of the musculoskeletal system and connective tissue (M00-M99)',
 '740-759:Congenital Anomalies:None':'Congenital malformations, deformations and chromosomal abnormalities (Q00-Q99)',
 '760-779:Certain Conditions Originating In The Perinatal Period:None':'Certain conditions originating in the perinatal period (P00-P96)',
 '780-799:Symptoms, Signs, And Ill-Defined Conditions:None': 'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified (R00-R99)',
 '800-999:Injury And Poisoning:None':'Injury, poisoning and certain other consequences of external causes (S00-T88)',
 'E000-E999:Supplementary Classification Of External Causes Of Injury And Poisoning:None':'External causes of morbidity (V00-Y99)',
 'V01-V91:Supplementary Classification Of Factors Influencing Health Status And Contact With Health Services:None': 'Factors influencing health status and contact with health services (Z00-Z99)',
 '001-139:Infectious And Parasitic Diseases:None': 'Certain infectious and parasitic diseases (A00-B99)'}


df_icd_9["chapter"] = df_icd_9["parent_node"].astype(str).map(Mapping)

final_processed_icd_codes = pd.concat([df_icd_9, df_icd_10])
final_processed_icd_codes = final_processed_icd_codes.drop('parent_node', axis=1)


