# Interview Preparation: Hotel & Resort Customer Reviews Analysis

## 1. Project Overview

**Problem Statement:** Analyze customer reviews to extract insights about hotel/resort performance, identify common complaints, detect sentiment patterns, and provide actionable recommendations for service improvement.

**Objective:** Build NLP pipeline for:
1. Sentiment analysis (positive/negative/neutral)
2. Topic modeling (identify common themes)
3. Review summarization
4. Rating prediction from text
5. Aspect-based sentiment (room, service, food, location)

**Business Value:** Improve customer satisfaction, prioritize improvements, competitive analysis, reputation management.

---

## 2. Technical Concepts

### Natural Language Processing
- **Sentiment Analysis:** Classify emotional tone of reviews
- **Topic Modeling:** Discover themes (cleanliness, staff, location)
- **Named Entity Recognition:** Extract hotel names, locations
- **Text Summarization:** Condense long reviews
- **Aspect-Based Sentiment:** Sentiment per aspect (room=positive, service=negative)

### Algorithms
- **Sentiment:** VADER, TextBlob, BERT
- **Topic Modeling:** LDA (Latent Dirichlet Allocation), NMF
- **Classification:** Naive Bayes, Logistic Regression, BERT
- **Clustering:** K-Means on embeddings

---

## 3. Mathematical Foundations

### TF-IDF
\[
\text{TF-IDF}(t, d) = \frac{f_{t,d}}{\sum_t f_{t,d}} \times \log\frac{N}{n_t}
\]

### LDA (Latent Dirichlet Allocation)
Probabilistic topic model:
\[
P(\text{word}|\text{document}) = \sum_{\text{topic}} P(\text{word}|\text{topic}) \times P(\text{topic}|\text{document})
\]

### Sentiment Score (VADER)
Compound score combining positive, negative, neutral:
\[
\text{compound} = \frac{\sum \text{valence}}{\sqrt{(\sum \text{valence})^2 + \alpha}}
\]
Range: [-1, 1]

### Cosine Similarity (Document Similarity)
\[
\text{similarity}(d_1, d_2) = \frac{d_1 \cdot d_2}{||d_1|| \times ||d_2||}
\]

---

## 4. Implementation Details

### Complete Analysis Pipeline

**1. Data Loading & Preprocessing**
```python
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Load reviews
df = pd.read_csv('hotel_reviews.csv')

# Preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return ' '.join(tokens)

df['processed_review'] = df['review_text'].apply(preprocess_text)
```

**2. Sentiment Analysis**
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Analyze sentiment
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['review_text'].apply(get_sentiment)
df['sentiment_score'] = df['review_text'].apply(
    lambda x: analyzer.polarity_scores(x)['compound']
)

# Sentiment distribution
print(df['sentiment'].value_counts())

# Sentiment by rating
import seaborn as sns
sns.boxplot(x='rating', y='sentiment_score', data=df)
```

**3. Topic Modeling (LDA)**
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Create document-term matrix
vectorizer = CountVectorizer(max_features=1000, max_df=0.8, min_df=2)
dtm = vectorizer.fit_transform(df['processed_review'])

# Train LDA
n_topics = 5
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(dtm)

# Display topics
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-10:]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")

# Output:
# Topic 0: room, clean, bed, comfortable, spacious
# Topic 1: staff, friendly, helpful, service, excellent
# Topic 2: food, breakfast, restaurant, delicious, buffet
# Topic 3: location, beach, near, convenient, walking
# Topic 4: pool, spa, facilities, gym, amenities
```

**4. Aspect-Based Sentiment**
```python
# Define aspects
aspects = {
    'room': ['room', 'bed', 'bathroom', 'shower', 'clean', 'spacious'],
    'service': ['staff', 'service', 'helpful', 'friendly', 'receptionist'],
    'food': ['food', 'breakfast', 'restaurant', 'dinner', 'meal'],
    'location': ['location', 'beach', 'downtown', 'convenient'],
    'amenities': ['pool', 'spa', 'gym', 'wifi', 'parking']
}

def aspect_sentiment(review, aspect_keywords):
    # Extract sentences mentioning aspect
    sentences = nltk.sent_tokenize(review)
    aspect_sentences = [s for s in sentences 
                       if any(kw in s.lower() for kw in aspect_keywords)]
    
    if not aspect_sentences:
        return None
    
    # Average sentiment of aspect sentences
    sentiments = [analyzer.polarity_scores(s)['compound'] 
                 for s in aspect_sentences]
    return np.mean(sentiments)

# Compute aspect sentiments
for aspect, keywords in aspects.items():
    df[f'{aspect}_sentiment'] = df['review_text'].apply(
        lambda x: aspect_sentiment(x, keywords)
    )

# Aggregate by hotel
hotel_aspects = df.groupby('hotel_name')[[f'{a}_sentiment' for a in aspects]].mean()
print(hotel_aspects)
```

**5. Rating Prediction from Text**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# TF-IDF features
tfidf = TfidfVectorizer(max_features=2000)
X = tfidf.fit_transform(df['processed_review'])
y = df['rating']  # 1-5 stars

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Predict
y_pred = lr.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score, mean_absolute_error
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} stars")
```

---

## 5. Outcomes & Results

### Typical Findings
- **Sentiment Distribution:** 60% positive, 25% neutral, 15% negative
- **Top Topics:** Room cleanliness, staff service, breakfast quality, location
- **Rating Prediction:** 70-75% accuracy, MAE ~0.5 stars

### Actionable Insights
1. **Strengths:** Location, friendly staff
2. **Weaknesses:** Breakfast variety, WiFi speed
3. **Improvement Priority:** Issues mentioned in negative reviews

---

## 6. Interview Questions & Answers

**Q1: What is VADER and why is it good for social media/reviews?**

**A1:**

**VADER (Valence Aware Dictionary and sEntiment Reasoner)**

**Advantages for Reviews:**
1. **Handles Social Media Language:** Emojis, slang, abbreviations
2. **Context-Aware:** "not good" correctly identified as negative
3. **Intensity:** Captures "very good" vs "good"
4. **Punctuation:** "Good!!!" more positive than "Good"
5. **No Training Needed:** Lexicon-based, works out-of-box

**Example:**
```python
analyzer.polarity_scores("The room was amazing!!!!")
# {'neg': 0.0, 'neu': 0.33, 'pos': 0.67, 'compound': 0.88}

analyzer.polarity_scores("The room was not good")
# {'neg': 0.43, 'neu': 0.57, 'pos': 0.0, 'compound': -0.48}
```

**Q2: How does LDA find topics?**

**A2:**

**LDA Assumptions:**
- Each document is mixture of topics
- Each topic is mixture of words

**Process:**
1. Initialize: Random topic assignments
2. For each word:
   - Probability of topic given document
   - Probability of word given topic
   - Reassign topic
3. Iterate until convergence

**Tuning n_topics:**
```python
# Coherence score to choose optimal topics
from gensim.models.coherencemodel import CoherenceModel

coherence_scores = []
for n in range(2, 11):
    lda = LatentDirichletAllocation(n_components=n)
    lda.fit(dtm)
    # Compute coherence
    # ...
    coherence_scores.append(score)

# Plot elbow curve
```

**Q3: What is aspect-based sentiment analysis?**

**A3:** **Fine-Grained Sentiment**

**Overall Sentiment:**
```
Review: "Great location but terrible service"
Sentiment: Neutral (pos + neg cancel out)
```

**Aspect-Based:**
```
Location: Positive
Service: Negative
```

**Value:** Pinpoint specific strengths/weaknesses

---

## Additional Resources

**Libraries:**
- VADER: vaderSentiment
- Topic Modeling: gensim, sklearn
- Advanced NLP: spaCy, transformers

**Papers:**
- Blei et al. (2003): "Latent Dirichlet Allocation"
- Hutto & Gilbert (2014): "VADER: A Parsimonious Rule-based Model"

