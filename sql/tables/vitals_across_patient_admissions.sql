/* uses tables chartevents and d_items from MIMIC-IV 2.0 database /*

-- Create vitals_across_patient_admissions
CREATE TABLE vitals_across_patient_admissions AS
SELECT v.subject_id, c.stay_id, c.itemid, c.hadm_id, c.charttime, c.storetime, c.value, c.valueuom
FROM chartevents c
LEFT JOIN patients_main v ON c.subject_id = v.subject_id
LEFT JOIN d_items_main d ON c.itemid = d.itemid
WHERE c.itemid IN (220045, 220210, 220179, 220277, 223761, 226104, 223762, 226329, 220739, 223900, 223901)
  AND d.itemid IN (220045, 220210, 220179, 220277, 223761, 226104, 226329, 220739, 223900, 223901, 223762);

-- Update itemid in vitals_across_patient_admissions with labels
UPDATE vitals_across_patient_admissions
SET itemid = (
    SELECT label
    FROM d_items
    WHERE vitals_across_patient_admissions.itemid = d_items_main.itemid
);

/* create refined table of vitals_across_patient_admissions for the eight vital measurements

-- Create refined_vitals_across_patient_admissions
CREATE TABLE refined_vitals_across_patient_admissions AS
SELECT subject_id, stay_id, hadm_id, charttime, gender, anchor_age,
       CASE
           WHEN itemid IN ('Temperature Fahrenheit', 'Blood Temperature CCO (C)', 'Temperature Celsius') THEN value
           ELSE NULL
       END AS temperature,
       CASE
           WHEN itemid = 'Respiratory Rate' THEN value
           ELSE NULL
       END AS respiration,
       CASE
           WHEN itemid = 'Heart Rate' THEN value
           ELSE NULL
       END AS heart_rate,
       CASE
           WHEN itemid = 'O2 saturation pulseoxymetry' THEN value
           ELSE NULL
       END AS sats,
       CASE
           WHEN itemid = 'Non Invasive Blood Pressure systolic' THEN value
           ELSE NULL
       END AS systolic_bp,
       CASE
           WHEN itemid = 'GCS - Verbal Response' THEN value
           ELSE NULL
       END AS gcs_verbal,
       CASE
           WHEN itemid = 'GCS - Motor Response' THEN value
           ELSE NULL
       END AS gcs_motor,
       CASE
           WHEN itemid = 'GCS - Eye Opening' THEN value
           ELSE NULL
       END AS gcs_eye
FROM vitals_across_patient_admissions;
