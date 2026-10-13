# NBA Over/Under Prediction

Machine Learning mini-project (UE24CS352A). Predicts the combined points scored in an NBA game and compares each model against the sportsbook Over/Under line. Based on the Stanford paper "The Bank is Open: AI in Sports Gambling".

## Models
Ridge regression, Random Forest, collaborative filtering (SVD), neural network, LSTM.

## Setup
Requires Python 3.9 or newer.

    pip install -r requirements.txt

## Data
Download `nba_2008-2026.csv` from the Kaggle dataset "NBA Betting Data" and place it in the project folder. Then convert it:

    python prepare_data.py nba_2008-2026.csv

This creates `data/logs.csv` (one row per team per game) and `data/odds.csv` (one row per game with the Over/Under line).

To test without the real data: `python synthetic.py`

## Run
Seasons are labeled by the year they end, so 2018 means the 2017-18 season.

Paper-style split (train 2012-13 to 2015-16, validate 2016-17, test 2017-18):

    python main.py

Recent seasons (train 2020-21 to 2023-24, validate 2024-25, test 2025-26):

    python main.py --train 2021,2022,2023,2024 --val 2025 --test 2026

The program prints each model's test MSE and how often it picks the correct side of the sportsbook line.

## Files
- prepare_data.py: converts the Kaggle file into the two CSVs
- features.py: last-3-game features, 10- and 20-game scoring averages, days of rest, season splits
- models.py: Ridge regression, Random Forest, collaborative filtering, neural network, LSTM
- main.py: trains all models and prints the results table
- synthetic.py: fake data generator for quick testing
- docs/: project write-up

## Results (test MSE; sportsbook line shown for comparison)
| Model | 2017-18 | 2025-26 |
|---|---|---|
| Ridge Regression | 347.35 | 367.16 |
| Random Forest | 351.73 | 372.82 |
| Collaborative Filtering | 418.23 | 423.75 |
| Neural Network | 352.30 | 381.40 |
| LSTM | 347.36 | 370.03 |
| Sportsbook line | 323.27 | 334.26 |
