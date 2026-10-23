# Data Directory

## Dataset: International Football Results

**Source:** Mart Jürisoo's International Football Results dataset  
**Repository:** https://github.com/martj42/international_results  
**Direct CSV URL:** https://raw.githubusercontent.com/martj42/international_results/master/results.csv

## Usage

The application will automatically attempt to download `results.csv` from the
URL above when first launched. The file is cached locally in this directory.

If the download fails (e.g., no internet connection), you can manually place a
`results.csv` file in this directory, or use the file-upload feature in the
Streamlit dashboard.

## Columns

| Column       | Type    | Description                              |
|--------------|---------|------------------------------------------|
| date         | date    | Match date (YYYY-MM-DD)                  |
| home_team    | string  | Name of the home team                    |
| away_team    | string  | Name of the away team                    |
| home_score   | integer | Goals scored by the home team            |
| away_score   | integer | Goals scored by the away team            |
| tournament   | string  | Tournament or competition name           |
| city         | string  | City where the match was played          |
| country      | string  | Country where the match was played       |
| neutral      | boolean | True if the venue was neutral            |

## Study Period

This project uses matches from **January 1, 2000 through December 31, 2025**.

## License / Attribution

The dataset is maintained by Mart Jürisoo and distributed under the terms of
the source repository. Please refer to the original repository for license
details before any redistribution.

## Note on Data Integrity

This project does not modify or fabricate any records in the source dataset.
All statistics are derived directly from the downloaded file. If the source
dataset contains errors, those will propagate into the analysis, which is
one of the documented limitations of this study.
