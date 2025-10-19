# Interview Preparation: Research Topic Prediction

## 1. Project Overview

**Problem Statement:** Classify research papers into topics/categories based on their abstracts and metadata using natural language processing and machine learning.

**Objective:** Build text classification models to automatically categorize academic papers, enabling better organization, search, and recommendation in digital libraries and databases.

**Applications:**
- Academic paper organization
- Automated tagging systems
- Research trend analysis
- Recommendation systems for researchers

---

## 2. Technical Concepts

### Text Classification
- **Multi-class Classification:** Assign paper to one of multiple research topics
- **NLP Preprocessing:** Tokenization, stopword removal, stemming/lemmatization
- **Feature Extraction:** TF-IDF, word embeddings

### Algorithms
- **Naive Bayes:** Probabilistic text classifier
- **Logistic Regression:** With TF-IDF features
- **Random Forest:** On text features
- **SVM:** Linear kernel for text
- **BERT/Transformers:** Deep learning (advanced)

---

## 3. Mathematical Foundations

### TF-IDF (Term Frequency-Inverse Document Frequency)

**Term Frequency:**
\[
TF(t, d) = \frac{\text{Count of term } t \text{ in document } d}{\text{Total terms in document } d}
\]

**Inverse Document Frequency:**
\[
IDF(t, D) = \log\frac{|D|}{|\{d \in D: t \in d\}|}
\]

**TF-IDF Score:**
\[
\text{TF-IDF}(t, d, D) = TF(t, d) \times IDF(t, D)
\]

**Intuition:**
- Common words (the, and) → Low IDF → Low TF-IDF
- Rare, discriminative words → High IDF → High TF-IDF

### Naive Bayes for Text
\[
P(\text{topic}|words) \propto P(\text{topic}) \prod_{i} P(word_i|\text{topic})
\]

### Cosine Similarity
\[
\text{similarity} = \frac{A \cdot B}{||A|| \times ||B||} = \frac{\sum A_i B_i}{\sqrt{\sum A_i^2}\sqrt{\sum B_i^2}}
\]

---

## 4. Implementation Details

### Workflow
```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load data
df = pd.read_csv('research_papers.csv')

# Text preprocessing
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    
    # Stem
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(w) for w in tokens]
    
    return ' '.join(tokens)

df['processed_abstract'] = df['abstract'].apply(preprocess_text)

# TF-IDF vectorization
tfidf = TfidfVectorizer(
    max_features=5000,  # Top 5000 words
    ngram_range=(1, 2),  # Unigrams and bigrams
    min_df=2,  # Ignore terms in <2 documents
    max_df=0.8  # Ignore terms in >80% documents
)

X = tfidf.fit_transform(df['processed_abstract'])
y = df['topic']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='saga'),
    'Linear SVM': LinearSVC(max_iter=1000)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

# Top features per topic
feature_names = tfidf.get_feature_names_out()
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

for i, topic in enumerate(lr.classes_):
    top_indices = np.argsort(lr.coef_[i])[-10:]
    top_words = [feature_names[idx] for idx in top_indices]
    print(f"\nTopic {topic}: {', '.join(top_words)}")
```

### Advanced: Word Embeddings
```python
from gensim.models import Word2Vec

# Train Word2Vec
tokenized = df['processed_abstract'].apply(lambda x: x.split())
w2v_model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=2)

# Document embedding: Average word vectors
def document_vector(text, model):
    words = text.split()
    word_vectors = [model.wv[w] for w in words if w in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(100)

X_w2v = np.array([document_vector(text, w2v_model) for text in df['processed_abstract']])
```

---

## 5. Outcomes & Results

### Typical Performance
- **Naive Bayes:** 75-85% accuracy
- **Logistic Regression:** 80-90% accuracy
- **Linear SVM:** 80-90% accuracy
- **BERT:** 90-95% accuracy (if used)

### Most Discriminative Features
- Domain-specific terms (e.g., "neural network" for CS, "genome" for biology)
- Methodology keywords (e.g., "qualitative", "quantitative")
- Statistical terms

---

## 6. Interview Questions & Answers

**Q1: Why use TF-IDF instead of simple word counts?**

**A1:**

**Word Counts Problem:**
- Common words (the, and, is) dominate
- Rare discriminative words underweighted

**TF-IDF Solution:**
- Down-weights common words (high document frequency)
- Up-weights rare, topic-specific words
- Better representation for classification

**Example:**
```
Word: "neural"
Appears in: 50/1000 CS papers, 1/1000 biology papers
→ High IDF, very discriminative for CS topic
```

**Q2: What is the purpose of stopword removal?**

**A2:**

**Stopwords:** Common words (the, is, and, in, of, ...)

**Remove Because:**
1. No discriminative power (appear in all topics)
2. Add noise to model
3. Increase dimensionality unnecessarily
4. Slow down training

**Example:**
```
Before: "the neural network is a machine learning algorithm"
After:  "neural network machine learning algorithm"
```

**Q3: Why use LinearSVC instead of SVC(kernel='linear')?**

**A3:**

**LinearSVC:**
- Optimized for linear kernel
- Uses liblinear library
- Faster for large datasets
- Scales better to high dimensions

**SVC(kernel='linear'):**
- General SVM implementation
- Uses libsvm library
- Slower but more flexible

**For Text (high-dimensional):** LinearSVC preferred

**Q4: How would you handle class imbalance in topic distribution?**

**A4:**

**Solutions:**
```python
# 1. Stratified split
train_test_split(..., stratify=y)

# 2. Class weights
LogisticRegression(class_weight='balanced')

# 3. Resampling
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_sm, y_sm = smote.fit_resample(X, y)

# 4. Evaluation metric
# Use F1-macro (average across classes)
# Instead of accuracy (biased toward majority)
```

**Q5: How would you improve this model?**

**A5:**

**1. Pre-trained Embeddings:**
```python
# Use Word2Vec, GloVe, or BERT embeddings
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Get contextualized embeddings
```

**2. Use Title + Abstract:**
```python
df['combined_text'] = df['title'] + ' ' + df['abstract']
```

**3. N-grams:**
```python
# Bigrams and trigrams
TfidfVectorizer(ngram_range=(1, 3))
```

**4. Deep Learning:**
```python
# LSTM or Transformer
from keras.layers import LSTM, Embedding

model = Sequential([
    Embedding(vocab_size, 128),
    LSTM(64),
    Dense(num_topics, activation='softmax')
])
```

**5. Hierarchical Classification:**
```python
# Level 1: Broad category (CS, Biology, Physics)
# Level 2: Specific topic (ML, Genetics, Quantum)
```

---

## Additional Resources

**Papers:**
- Blei et al. (2003): "Latent Dirichlet Allocation" (Topic Modeling)
- Devlin et al. (2018): "BERT: Pre-training of Deep Bidirectional Transformers"

**Datasets:**
- arXiv: Research paper metadata
- PubMed: Medical research papers
- ACL Anthology: NLP papers

