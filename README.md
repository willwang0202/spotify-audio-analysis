# Spotify Track Popularity & Audio Features

*A DSC 80 @ UCSD data science project by Will Wang*

What makes a song popular on Spotify? This project explores audio and metadata features for
**6,000 tracks** across six musically distinct genres — pop, hip-hop, rock, country, classical, and
jazz — to understand what separates a hit from an obscure track, and to build a model that predicts it.

## Introduction

The [Spotify Music Tracks dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
contains audio characteristics computed by Spotify (like `energy`, `danceability`, and `valence`)
alongside metadata such as `popularity`, `explicit`, and `release_date` for 114,000 tracks across 114
genres. I focus on **6,000 tracks (1,000 per genre)** from six contrasting genres so that comparisons
are meaningful.

**The question this project centers on:** *Which audio and metadata features are associated with how
popular a track is, and can we predict whether a track will be popular from those features?* Popularity
drives what gets recommended and heard, so it matters to listeners, artists, and labels alike. The most
relevant columns are `popularity` (the 0–100 score we study), `track_genre`, and audio features such as
`energy`, `danceability`, `valence`, `acousticness`, `loudness`, and `tempo`.

## Data Cleaning and Exploratory Data Analysis

**Cleaning steps:** I dropped a leftover index column, parsed the mixed-format `release_date` into a
numeric `release_year`, converted `duration_ms` to `duration_min`, and created the binary label
`is_popular = popularity >= 70`. I deliberately left `tempo`'s missing values as `NaN` because that
missingness is studied below. The head of the cleaned data:

| track_name             | track_genre   |   popularity | is_popular   |   energy |   danceability |   valence |   tempo |
|:-----------------------|:--------------|-------------:|:-------------|---------:|---------------:|----------:|--------:|
| Zara Zara              | classical     |           58 | False        |    0.268 |          0.643 |     0.62  | nan     |
| Kajra Re               | classical     |           59 | False        |    0.898 |          0.484 |     0.68  | nan     |
| Zara Zara - Lofi       | classical     |           54 | False        |    0.638 |          0.608 |     0.439 | 140.109 |
| Vaseegara              | classical     |           68 | False        |    0.293 |          0.695 |     0.637 | nan     |
| Zara Zara - LoFi Chill | classical     |           59 | False        |    0.308 |          0.583 |     0.241 | 118.226 |

**Univariate analysis.** Popularity is strongly right-skewed — most tracks score well below 50, and
genuinely popular tracks (≥ 70) are rare.

<iframe src="assets/popularity_dist.html" width="100%" height="480" frameborder="0"></iframe>

**Bivariate analysis.** Popularity differs sharply by genre: pop and hip-hop sit far above classical and
jazz, so genre is a likely confounder for any feature–popularity relationship.

<iframe src="assets/popularity_by_genre.html" width="100%" height="480" frameborder="0"></iframe>

**Interesting aggregates.** Grouping by genre confirms that pop and hip-hop dominate the "popular rate,"
and that explicit content is concentrated in hip-hop:

| track_genre   |   num_tracks |   mean_popularity |   popular_rate |   mean_energy |   explicit_rate |
|:--------------|-------------:|------------------:|---------------:|--------------:|----------------:|
| classical     |         1000 |            13.055 |          0.007 |         0.19  |           0     |
| jazz          |         1000 |            13.628 |          0.025 |         0.353 |           0.003 |
| country       |         1000 |            17.028 |          0.087 |         0.597 |           0.03  |
| rock          |         1000 |            19.001 |          0.198 |         0.679 |           0.043 |
| hip-hop       |         1000 |            37.759 |          0.134 |         0.683 |           0.319 |
| pop           |         1000 |            47.576 |          0.317 |         0.606 |           0.074 |

## Assessment of Missingness

**NMAR.** I do **not** believe any column is NMAR. The one column with substantial missingness is
`tempo` (~24% missing). Tempo is estimated by Spotify's algorithm — not self-reported — so whether it is
recorded depends on properties of the audio (a detectable beat), not on the unobserved tempo value
itself. To call it NMAR I would want Spotify's internal beat-detection confidence; absent that, the
evidence points to **MAR**.

**Missingness dependency.** Permutation tests show `tempo` missingness **depends on genre** (Total
Variation Distance = 0.18, p ≈ 0) — beat-ambiguous classical and jazz lose tempo far more often — and on
`energy` (p ≈ 0), but **not on `duration_ms`** (p ≈ 0.83). Missingness explained by observed columns is
the signature of MAR.

<iframe src="assets/tempo_missingness.html" width="100%" height="480" frameborder="0"></iframe>

## Hypothesis Testing

**Are explicit tracks more energetic than non-explicit tracks?**

- **Null:** explicit and non-explicit tracks have the same mean energy; differences are due to chance.
- **Alternative:** explicit tracks have *higher* mean energy.
- **Test statistic:** difference in mean energy (explicit − non-explicit); **α = 0.05**; permutation test
  with 1,000 shuffles.

The observed difference of **0.149** is larger than every value under the null, giving **p ≈ 0**, so
we **reject the null hypothesis**: explicit tracks tend to be more energetic in these genres. (This is
evidence of an association, not proof of causation — explicit tracks cluster in hip-hop and pop.)

<iframe src="assets/hypothesis_null.html" width="100%" height="480" frameborder="0"></iframe>

## Framing a Prediction Problem

I predict whether a track is **popular** (`popularity ≥ 70`) — a **binary classification** problem. The
classes are imbalanced (~13% popular), so a model that always predicts "not popular" would reach ~87%
accuracy while being useless. I therefore evaluate with the **F1-score on the popular class** (reporting
accuracy, precision, and recall too). I only use features available at release time — audio features and
metadata like genre, explicitness, duration, and release year — and exclude `popularity` itself to avoid
data leakage.

## Baseline Model

The baseline is a single scikit-learn `Pipeline` predicting `is_popular` from **three features**:
`track_genre` (nominal, one-hot encoded), `energy`, and `danceability` (quantitative, as-is) — 2
quantitative + 1 nominal. The classifier is a class-balanced `LogisticRegression`. On a held-out 25% test
set:

| Metric | Score |
| --- | --- |
| Accuracy | 0.628 |
| Precision | 0.222 |
| Recall | 0.760 |
| **F1 (popular class)** | **0.344** |

The model recovers most popular tracks (high recall) but at modest precision. It is a fair starting
point that clearly beats the degenerate majority-class predictor on F1, with room to improve.

## Final Model

*In progress.* I plan to add more audio features and engineered features (`release_year`,
`duration_min`), join artist-level follower counts from `artists.csv`, and tune a tree-based model
(random forest / gradient boosting) with `GridSearchCV`, optimizing F1 on the same train/test split.

## Fairness Analysis

*In progress.* I will test whether the final model's performance differs across groups (for example,
older vs. newer tracks) using a permutation test on the chosen evaluation metric.
