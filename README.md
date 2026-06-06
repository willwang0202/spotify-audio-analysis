# Spotify Track Popularity and Audio Features

**Name:** Will Wang

## Introduction

This project investigates the Spotify Music Tracks dataset, which contains audio and metadata features for 114,000 tracks across 114 genres. Each row describes one track, including audio features computed by Spotify's algorithms, such as `energy`, `danceability`, `valence`, and `acousticness`, along with metadata such as `popularity`, `explicit`, `duration_ms`, and `release_date`.

The central question for this project is:

**Which audio and metadata features are associated with how popular a track is, and can we predict whether a track will be popular from those features?**

This question matters because Spotify popularity affects what listeners see in playlists, recommendations, and search results. Popularity is not just a number attached to a track; it shapes the attention that artists and labels receive. To keep the comparisons interpretable, I focus on six musically distinct genres: `pop`, `hip-hop`, `rock`, `country`, `classical`, and `jazz`. This gives 6,000 tracks total, with 1,000 tracks per genre.

The columns most relevant to this analysis are:

| Column | Description |
|:--|:--|
| `popularity` | Spotify popularity score from 0 to 100. This is the basis for the prediction target. |
| `track_genre` | Spotify genre label. I use six selected genres. |
| `energy` | Perceptual intensity and activity, from 0 to 1. |
| `danceability` | How suitable the track is for dancing, from 0 to 1. |
| `valence` | Musical positiveness, from 0 to 1. |
| `acousticness` | Confidence that the track is acoustic, from 0 to 1. |
| `instrumentalness` | Probability that the track has no vocals, from 0 to 1. |
| `loudness` | Overall loudness in decibels. |
| `tempo` | Estimated beats per minute. This column has substantial missingness. |
| `explicit` | Whether the track contains explicit content. |
| `duration_ms` | Track length in milliseconds. |
| `release_date` | Release date in mixed formats such as `YYYY`, `YYYY-MM`, and `YYYY-MM-DD`. |

## Data Cleaning and Exploratory Data Analysis

The raw `music_tracks.csv` file is already mostly tidy, with one row per track. I made the following targeted cleaning changes:

1. Dropped `Unnamed: 0`, which is just a leftover CSV index.
2. Parsed `release_date` into a numeric `release_year` by extracting the first four digits.
3. Converted `duration_ms` into `duration_min`.
4. Created `is_popular`, a binary label equal to `True` when `popularity >= 70`.
5. Kept missing `tempo` values as `NaN`, because tempo missingness is analyzed directly in the missingness section.

Here is the head of the cleaned DataFrame, restricted to columns used in the analysis:

| track_name             | track_genre   |   popularity | is_popular   |   energy |   danceability |   valence |   tempo |   release_year |   duration_min |
|:-----------------------|:--------------|-------------:|:-------------|---------:|---------------:|----------:|--------:|---------------:|---------------:|
| Zara Zara              | classical     |           58 | False        |    0.268 |          0.643 |     0.620 |     nan |           2001 |          4.971 |
| Kajra Re               | classical     |           59 | False        |    0.898 |          0.484 |     0.680 |     nan |           2005 |          8.043 |
| Zara Zara - Lofi       | classical     |           54 | False        |    0.638 |          0.608 |     0.439 | 140.109 |           1984 |          3.657 |
| Vaseegara              | classical     |           68 | False        |    0.293 |          0.695 |     0.637 |     nan |           1972 |          4.986 |
| Zara Zara - LoFi Chill | classical     |           59 | False        |    0.308 |          0.583 |     0.241 | 118.226 |           1987 |          6.462 |

### Univariate Analysis

The popularity distribution is strongly right-skewed. Most tracks score well below 50, while tracks with `popularity >= 70` are rare. This is why the prediction problem later is imbalanced.

<iframe src="assets/popularity-distribution.html" width="800" height="500" frameBorder="0"></iframe>

### Bivariate Analysis

Popularity differs substantially by genre. `pop` and `hip-hop` have much higher popularity distributions than `classical` and `jazz`, which means genre is likely a confounder when studying relationships between audio features and popularity.

<iframe src="assets/popularity-by-genre.html" width="800" height="500" frameBorder="0"></iframe>

The relationship between `energy` and `acousticness` also varies by genre. Acoustic genres tend to cluster at high acousticness and lower energy, while `rock`, `hip-hop`, and `pop` are generally more energetic.

<iframe src="assets/energy-vs-acousticness.html" width="800" height="500" frameBorder="0"></iframe>

### Interesting Aggregates

| track_genre   |   num_tracks |   mean_popularity |   median_popularity |   popular_rate |   mean_energy |   explicit_rate |
|:--------------|-------------:|------------------:|--------------------:|---------------:|--------------:|----------------:|
| classical     |         1000 |            13.055 |                   3 |          0.007 |         0.190 |           0.000 |
| jazz          |         1000 |            13.628 |                   0 |          0.025 |         0.353 |           0.003 |
| country       |         1000 |            17.028 |                   0 |          0.087 |         0.597 |           0.030 |
| rock          |         1000 |            19.001 |                   0 |          0.198 |         0.679 |           0.043 |
| hip-hop       |         1000 |            37.759 |                  58 |          0.134 |         0.683 |           0.319 |
| pop           |         1000 |            47.576 |                  66 |          0.317 |         0.606 |           0.074 |

This table shows that `pop` has the highest popular-track rate, followed by `rock` and `hip-hop`. It also shows that explicit tracks are concentrated most heavily in `hip-hop`, which matters for the hypothesis test.

| track_genre   |   non_explicit |   explicit |
|:--------------|---------------:|-----------:|
| classical     |          0.190 |        nan |
| jazz          |          0.352 |      0.635 |
| country       |          0.594 |      0.672 |
| rock          |          0.675 |      0.780 |
| hip-hop       |          0.703 |      0.640 |
| pop           |          0.603 |      0.647 |

Within most genres that have explicit tracks, explicit tracks have higher mean energy than non-explicit tracks. This motivates the hypothesis test below.

## Assessment of Missingness

The column with non-trivial missingness is `tempo`: 23.5% of tracks in the selected subset are missing tempo. I do **not** believe `tempo` is NMAR. Spotify estimates tempo algorithmically from the audio, so missingness should depend on observable properties of the track, such as genre, energy, acousticness, or beat clarity, rather than directly on the unobserved true tempo value.

The missingness pattern is strongly related to genre:

<iframe src="assets/tempo-missingness-by-genre.html" width="800" height="500" frameBorder="0"></iframe>

To test missingness dependency, I used permutation tests:

- For `track_genre`, a categorical column, the test statistic is Total Variation Distance between the genre distribution when `tempo` is missing and when `tempo` is present.
- For numeric columns, the test statistic is the absolute difference in means between the `tempo` missing and non-missing groups.

For `track_genre`, the observed TVD was 0.176 and the simulated p-value was 0.000 across 1,000 permutations.

<iframe src="assets/tempo-missingness-permutation.html" width="800" height="500" frameBorder="0"></iframe>

For `energy`, the observed absolute difference in means was 0.139 with p-value 0.000, so `tempo` missingness depends on `energy`. For `duration_ms`, the observed absolute difference was 741.5 ms with p-value 0.826, so I fail to reject the null that `tempo` missingness is independent of duration. Overall, `tempo` is best described as MAR because its missingness is explainable using observed columns.

## Hypothesis Testing

I test whether explicit tracks are more energetic than non-explicit tracks among the six selected genres.

**Null hypothesis:** Explicit and non-explicit tracks have the same mean `energy`; any observed difference is due to random chance.

**Alternative hypothesis:** Explicit tracks have higher mean `energy` than non-explicit tracks.

**Test statistic:** `mean energy of explicit tracks - mean energy of non-explicit tracks`.

**Significance level:** 0.05.

| explicit   |   count |   mean |   median |   std |
|:-----------|--------:|-------:|---------:|------:|
| False      |    5531 |  0.506 |    0.534 | 0.263 |
| True       |     469 |  0.656 |    0.652 | 0.149 |

The observed difference in mean energy was 0.149. I used a one-sided permutation test by shuffling the `explicit` labels 1,000 times. The simulated p-value was 0.000.

<iframe src="assets/energy-hypothesis-permutation.html" width="800" height="500" frameBorder="0"></iframe>

Since the p-value is below 0.05, I reject the null hypothesis. The data provide evidence that explicit tracks tend to have higher energy than non-explicit tracks in these selected genres. This is evidence of association, not proof of causation, and the result may partly reflect the fact that explicit tracks are concentrated in genres like `hip-hop` and `pop`.

## Framing a Prediction Problem

The prediction problem is to predict whether a track is popular.

The response variable is `is_popular`, defined as:

- `True` if `popularity >= 70`
- `False` otherwise

This is a **binary classification** problem. I chose this target because predicting whether a track becomes highly popular is interpretable and connects directly to the project question.

| is_popular   |   proportion |   count |
|:-------------|-------------:|--------:|
| False        |        0.872 |    5232 |
| True         |        0.128 |     768 |

Because only 12.8% of the tracks are popular, accuracy alone would be misleading. A model that always predicts "not popular" would have high accuracy but would never identify popular tracks. For this reason, I use **F1-score on the popular class** as the primary metric, while also reporting accuracy, precision, and recall.

At prediction time, I use only track-level audio and metadata features that would be available without knowing the popularity outcome: `track_genre`, `explicit`, `duration_min`, `release_year`, `energy`, `danceability`, `valence`, `acousticness`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, and `tempo`. I exclude `popularity` itself because it defines the label.

## Baseline Model

The baseline model is a single sklearn `Pipeline` using three original features:

- `track_genre`: nominal, one-hot encoded.
- `energy`: quantitative, passed through unchanged.
- `danceability`: quantitative, passed through unchanged.

The classifier is `LogisticRegression` with `class_weight='balanced'` because the popular class is rare. The baseline has 2 quantitative features, 1 nominal feature, and 0 ordinal features.

| model                        |   accuracy |   precision |   recall |    f1 |
|:-----------------------------|-----------:|------------:|---------:|------:|
| Baseline Logistic Regression |      0.628 |       0.222 |    0.760 | 0.344 |

The baseline is not strong, but it is a meaningful starting point. It has high recall because it catches many popular tracks, but its precision is low, meaning it also incorrectly labels many non-popular tracks as popular.

## Final Model

The final model uses a random forest classifier inside a single sklearn `Pipeline`. It improves the baseline in three ways:

1. It uses more track-level features: `valence`, `acousticness`, `instrumentalness`, `loudness`, `speechiness`, `liveness`, `tempo`, `explicit`, `duration_min`, and `release_year`.
2. It engineers new features inside the pipeline:
   - `tempo_missing`
   - `release_decade`
   - `energy_x_danceability`
   - `acoustic_energy_gap`
3. It tunes hyperparameters using `GridSearchCV`, optimizing F1-score.

The tuned hyperparameters were:

```text
max_depth = None
min_samples_leaf = 15
n_estimators = 200
```

I did not use `artists.csv` in the final model. Joining on artist names is ambiguous for multi-artist tracks, and current artist popularity/followers would be difficult to justify at the time of prediction. Avoiding that join reduces leakage risk.

| model                        |   accuracy |   precision |   recall |    f1 |
|:-----------------------------|-----------:|------------:|---------:|------:|
| Baseline Logistic Regression |      0.628 |       0.222 |    0.760 | 0.344 |
| Final Random Forest          |      0.813 |       0.362 |    0.609 | 0.454 |

The final model improves F1-score from 0.344 to 0.454 on the same held-out test set. Its recall is lower than the baseline's, but its precision is much better, so the overall balance between precision and recall improves.

<iframe src="assets/final-confusion-matrix.html" width="800" height="500" frameBorder="0"></iframe>

## Fairness Analysis

I test whether the final model performs worse for acoustic/traditional genres than for mainstream/high-energy genres.

- **Group X:** acoustic/traditional genres: `classical`, `jazz`, `country`
- **Group Y:** mainstream/high-energy genres: `pop`, `hip-hop`, `rock`
- **Metric:** recall on the popular class
- **Null hypothesis:** The model is fair. Recall is roughly the same for both groups, and any observed difference is due to random chance.
- **Alternative hypothesis:** The model is unfair. Recall is lower for acoustic/traditional genres.
- **Test statistic:** `recall(mainstream/high-energy) - recall(acoustic/traditional)`
- **Significance level:** 0.05

| genre_group            |   n_tracks |   n_popular_actual |   precision |   recall |   f1 |
|:-----------------------|-----------:|-------------------:|------------:|---------:|-----:|
| acoustic_traditional   |        758 |                 32 |       0.000 |    0.000 | 0.000 |
| mainstream_high_energy |        742 |                160 |       0.368 |    0.731 | 0.490 |

The observed recall gap was 0.731. I then permuted group labels in the held-out test set 1,000 times while keeping the final model fixed. The simulated p-value was 0.000.

<iframe src="assets/fairness-recall-gap-permutation.html" width="800" height="500" frameBorder="0"></iframe>

Because the p-value is below 0.05, I reject the null hypothesis. The final model shows evidence of unfair performance across these genre groups: it misses popular acoustic/traditional tracks much more often than popular mainstream/high-energy tracks. This likely happens because the popular class is much rarer in the acoustic/traditional group, giving the model fewer positive examples to learn from.
