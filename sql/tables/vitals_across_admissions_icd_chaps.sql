/* add another column to vitals_across_admissions_icd table which specifies
  which broader disease chapter the icd code belongs to using icd_chapter_hierarchy.csv
  found in data /*

/*-- Create vitals_across_admissions_icd_chaps

 */
CREATE TABLE vitals_across_admissions_icd_chaps AS
SELECT v.*, i.icd_chapter
FROM vitals_across_admissions_icd v
JOIN icd_chapter_hierarchy i
ON i.icd_code = v.icd_code;
