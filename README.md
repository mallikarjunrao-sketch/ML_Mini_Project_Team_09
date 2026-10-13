# NBA Over/Under Prediction

Machine Learning mini-project, UE24CS352A (PES University).

## Team
| Name | SRN |
|---|---|
| Nishitha S | PES1UG24AM180 |
| Mallikarjuna Rao R.V | PES1UG24AM155 |

## Overview
Before every NBA game, sportsbooks publish an Over/Under line: their estimate of the combined points both teams will score. A bettor wins by guessing correctly whether the real total ends above (Over) or below (Under) that line.

This project predicts the total points of each game using only information available before tip-off, compares five machine learning models against the sportsbook's own line, and measures how often each model would pick the correct side of the line.

The project follows the approach of Bucquet and Sarukkai, "The Bank is Open: AI in Sports Gambling" (Stanford CS229), adapted to a different dataset. See "Relation to the original repository" below.

## Dataset
Kaggle "NBA Betting Data" file (`nba_2008-2026.csv`), with one row per game: final scores, quarter and overtime scores, and the closing Over/Under line (`total`).

- 24,440 games from 2007-08 to 2025-26; 22,808 regular-season games are used (playoffs are excluded).
- The Over/Under line is used only to evaluate the models. It is never an input feature.
- Seasons are labeled by the year they end, so 2018 means the 2017-18 season.

The file has no box-score statistics (rebounds, assists, steals) and no travel distance, which the original paper used. Scoring-based features are used instead.

## Approach
1. **Prepare data** (`prepare_data.py`): keep regular-season games, create one row per team per game, and compute days of rest from the gaps between games (capped at 7 days).
2. **Build features** (`features.py`): for every game, use only earlier games, so no future information leaks in.
3. **Split by season** so training always comes before validation, and validation before test.
4. **Train five models** to predict the total points, minimising mean squared error (MSE).
5. **Evaluate** on the test season: MSE, and Beats %.

### Features (94 inputs per game)
- Each team's last 3 games: points scored, points allowed, points in each quarter, and overtime points.
- For each of those games, the opponent's season-to-date averages of the same statistics (a measure of opponent strength).
- Each team's average points scored and allowed over its last 10 and last 20 games (added in this project; a longer history is less noisy than 3 games).
- Days of rest for each team.

That is 28 values per game step (both teams) over 3 steps, plus 10 extra values.

### Models
| Model | Description |
|---|---|
| Ridge regression | Linear model with a penalty on large weights (strength 1000) |
| Random Forest | 300 decision trees, at least 5 games per leaf |
| Collaborative filtering | Home-team by away-team matrix of totals from the previous 5 days, completed with a rank-5 SVD (the idea behind movie recommendations) |
| Neural network | Fully connected layers 500, 100, 20, 1 with ReLU, Adam optimiser, early stopping on validation loss |
| LSTM | One layer, 20 hidden units, dropout 0.2, reads the last 3 games in order, early stopping on validation loss |

Inputs are standardised using training data only.

## Setup
Requires Python 3.9 or newer.

    pip install -r requirements.txt

## Data preparation
Download `nba_2008-2026.csv` from the Kaggle dataset "NBA Betting Data" and place it in the project folder. Then run:

    python prepare_data.py nba_2008-2026.csv

This creates `data/logs.csv` (one row per team per game) and `data/odds.csv` (one row per game with the Over/Under line).

To test the code without the real data, run `python synthetic.py` instead. It writes fake data into the same two files, so run `prepare_data.py` again before producing real results.

## Run
Paper-style split (train 2012-13 to 2015-16, validate 2016-17, test 2017-18):

    python main.py

Recent seasons (train 2020-21 to 2023-24, validate 2024-25, test 2025-26):

    python main.py --train 2021,2022,2023,2024 --val 2025 --test 2026

Expected output: the number of games in each split, then a table with each model's test MSE and Beats %. The neural network and LSTM start from random weights, so their numbers can differ slightly between runs.

## Metrics
- **MSE (mean squared error):** the average of (predicted total minus real total) squared. Lower is better.
- **Beats %:** how often the model picks the correct side of the sportsbook line. If the model predicts above the line it is betting Over; below, Under. A game counts as correct when that matches whether the real total ended above or below the line. Games that land exactly on the line are skipped. 50% is a coin flip, and the paper says about 52-53% is needed to profit after betting fees.

## Results
Test MSE (lower is better) and Beats %, with the sportsbook line for comparison.

| Model | MSE 2017-18 | Beats % | MSE 2025-26 | Beats % |
|---|---|---|---|---|
| Ridge Regression | 347.35 | 49.2 | 367.16 | 49.3 |
| Random Forest | 351.73 | 49.5 | 372.82 | 49.2 |
| Collaborative Filtering | 418.23 | 49.6 | 423.75 | 50.4 |
| Neural Network | 352.30 | 50.5 | 381.40 | 50.2 |
| LSTM | 347.36 | 50.6 | 370.03 | 50.9 |
| Sportsbook line | 323.27 | - | 334.26 | - |

### Findings
- The sportsbook line is more accurate than every model on both test seasons.
- Ridge regression and the LSTM are our best models, with the neural network and Random Forest close behind. The differences among them are small.
- Adding 10- and 20-game averages cut the error by roughly 10% compared with using only the last 3 games.
- All models pick the correct side of the line only 49-51% of the time, which is chance level. No reliable betting edge was found, which agrees with the paper.
- Errors are higher on 2025-26 for every model, including the sportsbook, but the ranking stays the same.

## Limitations
- No injuries, lineups, player-level or box-score data, which the sportsbook clearly uses.
- One test season per split and no confidence intervals, so small differences between models are not meaningful.
- Hyperparameters were not tuned exhaustively.
- Neural network results vary slightly between runs.

## Project structure
    nba_ou/
    |-- prepare_data.py   converts the Kaggle file into the two CSVs
    |-- features.py       builds the features and the season splits
    |-- models.py         Ridge, Random Forest, collaborative filtering, neural network, LSTM
    |-- main.py           trains all models and prints the results table
    |-- synthetic.py      fake data generator for quick testing
    |-- requirements.txt  Python packages
    |-- docs/             project write-up (PDF)
    `-- data/             generated CSV files (not stored in the repository)

## Troubleshooting
- `No such file or directory: requirements.txt` or `main.py`: open the terminal inside the folder that contains those files.
- `ModuleNotFoundError`: run `pip install -r requirements.txt` again.
- `FileNotFoundError: data/logs.csv`: run `python prepare_data.py nba_2008-2026.csv` first, with the Kaggle file in the project folder.
- Results look unrealistic: you may have run `synthetic.py` last; run `prepare_data.py` again.

## Relation to the original repository
The authors' repository is a set of notebooks built on box-score data scraped from Basketball Reference (2007-2018) and a city-distance table. Our dataset is a Kaggle file with scores, quarter scores and the Over/Under line up to 2025-26, so the data preparation and features were rewritten. Kept from the paper: the same model families, the same season-based split and MSE as the main measure. Added here: Ridge regression, 10- and 20-game scoring averages, and a second test on recent seasons.

## References
- Bucquet, A. and Sarukkai, V. The Bank is Open: AI in Sports Gambling. Stanford CS229 project report.
- Authors' code: https://github.com/abucquet/bank_is_open
- Kaggle, NBA Betting Data (scores and Over/Under lines, 2007-2026).
