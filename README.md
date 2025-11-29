# Sentiment Analysis of Amazon Alexa Reviews

This repository contains a sentiment analysis project built on the `amazon_alexa.tsv` dataset. The objective is to classify verified customer reviews as positive or negative using classical machine learning models (Random Forest, XGBoost) following a standard NLP preprocessing pipeline. The project includes exploratory data analysis, feature engineering, text preprocessing, model training, and evaluation.

## Project Overview

The analysis follows these major steps:

1. **Data loading and inspection**
   - Read the tab-separated dataset `amazon_alexa.tsv`.
   - Basic checks for shape, missing values and data types.
   - Remove rows with missing review text.

2. **Exploratory data analysis (EDA)**
   - Inspect distribution of ratings and feedback (binary sentiment label).
   - Examine product variations and their average ratings.
   - Compute and visualise review length distributions.
   - Generate word clouds for overall reviews, and separately for unique words appearing in positive and negative reviews.

3. **Text preprocessing**
   - Lowercasing and removal of non-alphabetic characters.
   - Stopword removal using NLTK’s English stopwords.
   - Porter stemming to reduce words to their root forms.
   - Construction of a cleaned corpus.

4. **Feature extraction**
   - Convert text to bag-of-words vectors using `CountVectorizer(max_features=2500)`.
   - Resulting feature matrix `X` has shape `(3149, 2500)`.

5. **Train / test split and scaling**
   - Split data into training and test sets with `test_size=0.3` and `random_state=15`.
   - Since the bag-of-words matrix contains counts, apply `MinMaxScaler` to scale features prior to tree-based models.

6. **Modeling**
   - Train a `RandomForestClassifier` on the scaled training set.
   - Train an `XGBClassifier` on the same features.
   - Evaluate both models on training and test data.

7. **Evaluation**
   - Report training and testing accuracy for both models.
   - Compute and display confusion matrices.
   - Typical performance observed in this experiment:
     - Random Forest
       - Training accuracy: 0.9946
       - Testing accuracy: 0.9429
     - XGBoost
       - Training accuracy: 0.9714
       - Testing accuracy: 0.9418
    
## Dataset

- File: `amazon_alexa.tsv` (tab-separated values)
- Typical columns:
  - `rating` (1–5)
  - `date`
  - `variation` (product variant)
  - `verified_reviews` (review text)
  - `feedback` (binary sentiment label: 1 = positive, 0 = negative)

Note: One row in the original dataset contained a missing `verified_reviews` value and is dropped in preprocessing.

## Key Implementation Details

- **Stopwords**: NLTK stopwords are used (`nltk.download('stopwords')`).
- **Stemming**: Porter stemmer (`nltk.stem.porter.PorterStemmer`) is applied.
- **Vectorization**: `CountVectorizer(stop_words='english', max_features=2500)` is used to create the bag-of-words representation.
- **Scaling**: `MinMaxScaler` is applied to the train and test feature matrices, which is sometimes useful even for tree-based models to maintain consistent numeric ranges.
- **Train/Test split**: `train_test_split(..., test_size=0.3, random_state=15)`.
- **Models**:
  - `RandomForestClassifier()` (default hyperparameters in the notebook)
  - `XGBClassifier()` (default hyperparameters in the notebook)

## Requirements

Install the Python packages used in the notebook:

```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn wordcloud xgboost
Additionally, download NLTK stopwords (within Python):
import nltk
nltk.download('stopwords')

Run the Jupyter notebook or Python script.
Execute all sections sequentially:
- Data loading and cleaning
- EDA and visualisations (histograms, word clouds)
- Text preprocessing and bag-of-words creation
- Train/test split and scaling
- Train Random Forest and XGBoost models
- Evaluate and display confusion matrices

## Observations & Notes
- The dataset is highly imbalanced. Accuracy alone is not sufficient; precision, recall, and F1-score should be monitored.
- Positive reviews are often short and enthusiastic; negative reviews tend to contain specific complaints.
- Very high training accuracy from Random Forest indicates possible overfitting.
- Improvements for research-quality modeling include:
  - Using TF-IDF vectorization
  - Applying SMOTE or class weighting
  - Performing hyperparameter tuning (GridSearchCV, StratifiedKFold)
  - Use SHAP or feature importance to interpret model decisions.
  - Evaluating neural approaches such as LSTM or transformer-based models
