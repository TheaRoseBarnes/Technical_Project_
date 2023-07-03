
/*
 table which finds the average of each patients vital measurements
 for duration of stay
 */

CREATE TABLE avg_vital_across_admissiomn AS
SELECT stay_id, subject_id, hadm_id,
       AVG(anchor_age),
       MIN(gender),
       AVG(temperature),
       AVG(respiration),
       AVG(heart_rate),
       AVG(sats),
       AVG(systolic_bp),
       MAX(gcs_verbal),
       MAX(gcs_motor),
       MAX(gcs_eye)

       FROM vitals_across_patient_admissions
GROUP BY stay_id;
