-- create table of vital mesaurements and icd codes for the prediction pipeline

CREATE TABLE prediction_data AS
SELECT subject_id,hadm_id,stay_id,charttime,
       AVG(anchor_age),
       MIN(gender),
       AVG(temperature),
       AVG(respiration),
       AVG(heart_rate),
       AVG(sats),
       AVG(systolic_bp),
        MAX(gcs_motor),
        MAX(gcs_verbal),
        MAX(gcs_eye),
        GROUP_CONCAT(icd_chapter),
        GROUP_CONCAT(icd_code),
        MAX(icd_code)
 FROM vitals_across_admissions_icd_chaps
GROUP BY strftime('%H', charttime, '+30 minutes'), stay_id;
