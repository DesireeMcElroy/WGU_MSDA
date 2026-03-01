WITH patient_cleaned AS (
  SELECT 
    patient_id AS "Patient ID"
  , lat AS "Latitude"
  , lng AS "Longitude"
  , population AS "Population"
  , children AS "Children"
  , age AS "Age"
  , gender AS "Gender"
  , marital AS "Marital Status"
  , income AS "Income"
  , readmis AS "Readmission"
  , initial_days AS "Length of Stay (days)"
  , totalcharge AS "Total Charge"
  , additional_charges AS "Additional Charges"
  , vitd_levels AS "Vitamin D Levels"
  , doc_visits AS "Doctor Visits"
  , full_meals AS "Full Meals Eaten"
  , vitd_supp AS "Vitamin D Supplement"
  , soft_drink AS "Soft Drink"
  , hignblood AS "High Blood Pressure"
  , stroke AS "Stroke"
  , job_id
  , admis_id
  , compl_id
  , location_id
  FROM patient
),

job AS (
  SELECT 
    job_id
  , job_title AS "Job Title"
  FROM job
),

admission AS (
  SELECT 
    admins_id
  , initial_admission AS "Initial Admission"
  FROM admission
),

complication AS (
  SELECT 
    complication_id
  , complication_risk AS "Complication Risk"
  FROM complication
),

location AS (
  SELECT 
    location_id
  , city AS "City"
  , state AS "State"
  , zip AS "ZIP"
  , county AS "County"
  FROM location
),

services AS (
  SELECT 
    patient_id
  , services AS "Services"
  , overweight AS "Overweight"
  , arthritis AS "Arthritis"
  , diabetes AS "Diabetes"
  , hyperlipidemia AS "Hyperlipidemia"
  , backpain AS "Back Pain"
  , anxiety AS "Anxiety"
  , allergic_rhinitis AS "Allergic Rhinitis"
  , reflux_esophagitis AS "Reflux Esophagitis"
  , asthma AS "Asthma"
  FROM servicesaddon
),

survey AS (
  SELECT 
    patient_id
  , item1 AS "Survey Item 1"
  , item2 AS "Survey Item 2"
  , item3 AS "Survey Item 3"
  , item4 AS "Survey Item 4"
  , item5 AS "Survey Item 5"
  , item6 AS "Survey Item 6"
  , item7 AS "Survey Item 7"
  , item8 AS "Survey Item 8"
  FROM survey_responses_addon
)

SELECT 
    p."Patient ID"
  , p."Latitude"
  , p."Longitude"
  , p."Population"
  , p."Children"
  , p."Age"
  , p."Gender"
  , p."Marital Status"
  , p."Readmission"
  , p."Length of Stay (days)"

  , ROUND(p."Income", 0) AS "Income"
  , ROUND(p."Total Charge", 2) AS "Total Charge"
  , ROUND(p."Additional Charges", 2) AS "Additional Charges"
  , ROUND(p."Vitamin D Levels", 4) AS "Vitamin D Levels"

  , p."Doctor Visits"
  , p."Full Meals Eaten"
  , p."Vitamin D Supplement"
  , p."Soft Drink"
  , p."High Blood Pressure"
  , p."Stroke"
  , j."Job Title"
  , a."Initial Admission"
  , c."Complication Risk"
  , l."City"

  , CASE l."State"
        WHEN 'AL' THEN 'Alabama'
        WHEN 'AK' THEN 'Alaska'
        WHEN 'AZ' THEN 'Arizona'
        WHEN 'AR' THEN 'Arkansas'
        WHEN 'CA' THEN 'California'
        WHEN 'CO' THEN 'Colorado'
        WHEN 'DC' THEN 'District of Columbia'
        WHEN 'CT' THEN 'Connecticut'
        WHEN 'DE' THEN 'Delaware'
        WHEN 'FL' THEN 'Florida'
        WHEN 'GA' THEN 'Georgia'
        WHEN 'HI' THEN 'Hawaii'
        WHEN 'ID' THEN 'Idaho'
        WHEN 'IL' THEN 'Illinois'
        WHEN 'IN' THEN 'Indiana'
        WHEN 'IA' THEN 'Iowa'
        WHEN 'KS' THEN 'Kansas'
        WHEN 'KY' THEN 'Kentucky'
        WHEN 'LA' THEN 'Louisiana'
        WHEN 'ME' THEN 'Maine'
        WHEN 'MD' THEN 'Maryland'
        WHEN 'MA' THEN 'Massachusetts'
        WHEN 'MI' THEN 'Michigan'
        WHEN 'MN' THEN 'Minnesota'
        WHEN 'MS' THEN 'Mississippi'
        WHEN 'MO' THEN 'Missouri'
        WHEN 'MT' THEN 'Montana'
        WHEN 'NE' THEN 'Nebraska'
        WHEN 'NV' THEN 'Nevada'
        WHEN 'NH' THEN 'New Hampshire'
        WHEN 'NJ' THEN 'New Jersey'
        WHEN 'NM' THEN 'New Mexico'
        WHEN 'NY' THEN 'New York'
        WHEN 'NC' THEN 'North Carolina'
        WHEN 'ND' THEN 'North Dakota'
        WHEN 'OH' THEN 'Ohio'
        WHEN 'OK' THEN 'Oklahoma'
        WHEN 'OR' THEN 'Oregon'
        WHEN 'PA' THEN 'Pennsylvania'
        WHEN 'PR' THEN 'Puerto Rico'
        WHEN 'RI' THEN 'Rhode Island'
        WHEN 'SC' THEN 'South Carolina'
        WHEN 'SD' THEN 'South Dakota'
        WHEN 'TN' THEN 'Tennessee'
        WHEN 'TX' THEN 'Texas'
        WHEN 'UT' THEN 'Utah'
        WHEN 'VT' THEN 'Vermont'
        WHEN 'VA' THEN 'Virginia'
        WHEN 'WA' THEN 'Washington'
        WHEN 'WV' THEN 'West Virginia'
        WHEN 'WI' THEN 'Wisconsin'
        WHEN 'WY' THEN 'Wyoming'
        ELSE NULL
        END AS "State Name"

  , l."ZIP"
  , l."County"
  , s."Services"
  , s."Overweight"
  , s."Arthritis"
  , s."Diabetes"
  , s."Hyperlipidemia"
  , s."Back Pain"
  , s."Anxiety"
  , s."Allergic Rhinitis"
  , s."Reflux Esophagitis"
  , s."Asthma"
  , sr."Survey Item 1"
  , sr."Survey Item 2"
  , sr."Survey Item 3"
  , sr."Survey Item 4"
  , sr."Survey Item 5"
  , sr."Survey Item 6"
  , sr."Survey Item 7"
  , sr."Survey Item 8"
FROM patient_cleaned p
LEFT JOIN job j ON p.job_id = j.job_id
LEFT JOIN admission a ON p.admis_id = a.admins_id
LEFT JOIN complication c ON p.compl_id = c.complication_id
LEFT JOIN location l ON p.location_id = l.location_id
LEFT JOIN services s ON p."Patient ID" = s.patient_id
LEFT JOIN survey sr ON p."Patient ID" = sr.patient_id;
