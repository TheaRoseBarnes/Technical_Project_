/* use diagnoses_icd from MIMIC-IV 2.0 and refined_vitals_across_patient_admissions
  to produce a table that contains vitals across patient admissions and icd disease codes /*

/* -- Create vitals_across_admissions_icd

 */
CREATE TABLE vitals_across_admissions_icd AS
SELECT v.*,
       di.icd_code, di.icd_version
FROM refined_vitals_across_patient_admissions v
JOIN diagnoses_icd di ON v.hadm_id = di.hadm_id;
