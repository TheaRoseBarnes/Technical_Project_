/* add another column to vitals_across_admissions_icd table which specifies
  which broader disease chapter the icd code belongs to using icd_chapter_hierarchy.csv 
  found in data /*
  
-- Create vitals_across_admissions_icd_chaps
CREATE TABLE vitals_across_admissions_icd_chaps AS
SELECT *
FROM vitals_across_admissions_icd
JOIN icd_chapter_hierarchy
ON icd_chapter_hierarchy.icd_code = vitals_across_admissions_icd.icd_code
