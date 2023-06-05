/* use diagnoses_icd from MIMIC-IV 2.0 and refined_vitals_across_patient_admissions
  to produce a table that contains vitals across patient admissions and icd disease codes /*

-- Create vitals_across_admissions_icd
CREATE TABLE vitals_across_admissions_icd AS
SELECT *
FROM refined_vitals_across_patient_admissions
JOIN diagnoses_icd_main di ON refined_vitals_across_patient_admissions.hadm_id = di.hadm_id;
