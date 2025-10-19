# Interview Preparation: Medical Chatbot (Meddy)

## 1. Project Overview

**Problem Statement:** Build an intelligent medical chatbot that understands natural language symptom descriptions, extracts symptoms, and predicts potential diseases with precautionary measures.

**Objective:** Create an end-to-end NLP system combining:
1. **Symptom Extraction:** Neural network to identify symptoms from user text
2. **Disease Prediction:** ML classifier to predict disease from symptoms
3. **Medical Information:** Provide disease descriptions and precautions

**Use Cases:**
- Preliminary medical assessment
- Healthcare accessibility in remote areas
- Triage support for medical professionals
- Health education and awareness

**Important Disclaimer:** Not a replacement for professional medical advice.

---

## 2. Technical Concepts

### Natural Language Processing (NLP)
- **Tokenization:** Breaking text into words
- **Stemming:** Reducing words to root form (running → run)
- **Bag of Words:** Representing text as word frequency vector
- **Intent Classification:** Identifying symptom from user sentence

### Two-Stage Architecture
1. **NLP Model (PyTorch):** Text → Symptom extraction
2. **ML Model (Sklearn):** Symptoms → Disease prediction

### Medical Knowledge Base
- **Symptom Dictionary:** 132 symptoms
- **Disease Database:** Multiple diseases with descriptions
- **Severity Scores:** Weight indicating symptom severity
- **Precautions:** Recommended preventive measures

---

## 3. Libraries & Technologies

### Core Libraries
- **PyTorch:** Deep learning for NLP model
- **NLTK:** Natural language processing utilities
- **Scikit-learn:** Disease prediction model
- **Flask:** Web framework for deployment
- **Pandas:** Data manipulation
- **NumPy:** Numerical operations
- **Pickle:** Model serialization

### Model Files
```
models/data.pth                    # PyTorch NLP model
models/fitted_model.pickle2        # Disease prediction model
data/symptom_Description.csv       # Disease descriptions
data/symptom_precaution.csv        # Precautionary measures
data/Symptom-severity.csv          # Severity weights
data/list_of_symptoms.pickle       # Complete symptom list
intents.json                       # Training data for NLP model
```

---

## 4. Code Architecture & Design Patterns

### File Structure
```
Medical Chatbot [END 2 END] [NLP]/
├── app.py                     # Flask application (main)
├── nnet.py                    # Neural network architecture
├── nltk_utils.py              # NLP utility functions
├── Meddy.ipynb                # Training notebook
├── intents.json               # Training data
├── models/
│   ├── data.pth              # Trained NLP model
│   └── fitted_model.pickle2   # Disease prediction model
├── data/
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   ├── Symptom-severity.csv
│   ├── dataset.csv
│   └── list_of_symptoms.pickle
├── static/                    # Frontend assets
└── templates/
    └── index.html            # Chat interface
```

### Design Patterns

**1. Pipeline Pattern**
```
User Input → Tokenization → Bag of Words → 
NLP Model → Symptom → Collect Symptoms → 
Disease Prediction → Information Retrieval → Response
```

**2. State Management**
```python
user_symptoms = set()  # Track symptoms per session
```

**3. Model Factory**
```python
# Load and initialize models at startup
nlp_model = NeuralNet(input_size, hidden_size, output_size)
nlp_model.load_state_dict(model_state)
nlp_model.eval()
```

---

## 5. Mathematical Foundations

### Neural Network (Feed-Forward)

**Architecture:**
```
Input Layer (vocab size) → 
Hidden Layer 1 (hidden_size) → ReLU → 
Hidden Layer 2 (hidden_size) → ReLU → 
Output Layer (num_symptoms) → Softmax
```

**Forward Pass:**
\[
h_1 = \text{ReLU}(W_1 x + b_1)
\]
\[
h_2 = \text{ReLU}(W_2 h_1 + b_2)
\]
\[
\hat{y} = W_3 h_2 + b_3
\]

### ReLU Activation
\[
\text{ReLU}(x) = \max(0, x)
\]

### Softmax (Output)
\[
P(\text{symptom}_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
\]

### Cross-Entropy Loss
\[
L = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)
\]

### Bag of Words Representation
For vocabulary size \(V\) and sentence \(s\):
\[
\text{BoW}(s) = [b_1, b_2, ..., b_V]
\]
where \(b_i = 1\) if word \(i\) in sentence, else 0.

### Porter Stemmer
Rule-based algorithm to remove suffixes:
- "running" → "run"
- "coughing" → "cough"
- "headaches" → "headach"

### Disease Prediction
Given symptom vector \(x \in \{0,1\}^{132}\):
\[
\text{disease} = \arg\max_d P(d | x)
\]

### Severity Score
Average and maximum severity:
\[
\text{avg\_severity} = \frac{1}{N}\sum_{i=1}^{N} \text{weight}(symptom_i)
\]
Alert if avg > 4 or max > 5.

---

## 6. Implementation Details

### Step-by-Step Code Walkthrough

**1. Model Loading (Startup)**
```python
import torch
import pickle
from nnet import NeuralNet

# Load NLP model
device = torch.device('cpu')
model_data = torch.load("models/data.pth")

input_size = model_data['input_size']
hidden_size = model_data['hidden_size']
output_size = model_data['output_size']
all_words = model_data['all_words']  # Vocabulary
tags = model_data['tags']            # Symptom list

nlp_model = NeuralNet(input_size, hidden_size, output_size).to(device)
nlp_model.load_state_dict(model_data['model_state'])
nlp_model.eval()  # Set to evaluation mode

# Load disease prediction model
with open('models/fitted_model.pickle2', 'rb') as f:
    prediction_model = pickle.load(f)

# Load medical knowledge base
diseases_description = pd.read_csv("data/symptom_Description.csv")
disease_precaution = pd.read_csv("data/symptom_precaution.csv")
symptom_severity = pd.read_csv("data/Symptom-severity.csv")
```

**2. Symptom Extraction from Text**
```python
from nltk_utils import bag_of_words
import nltk

def get_symptom(sentence):
    # Tokenize
    sentence = nltk.word_tokenize(sentence)  # "I have a headache" → ["I", "have", "a", "headache"]
    
    # Convert to bag of words
    X = bag_of_words(sentence, all_words)  # [0, 0, 1, 0, ..., 0] (one-hot encoding)
    X = X.reshape(1, X.shape[0])  # Add batch dimension
    X = torch.from_numpy(X)
    
    # Predict symptom
    output = nlp_model(X)  # Forward pass
    _, predicted = torch.max(output, dim=1)  # Get argmax
    tag = tags[predicted.item()]  # Map to symptom name
    
    # Get confidence
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()].item()
    
    return tag, prob
```

**3. Bag of Words Implementation**
```python
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

def bag_of_words(tokenized_sentence, all_words):
    # Stem each word
    tokenized_sentence = [stemmer.stem(w.lower()) for w in tokenized_sentence]
    
    # Initialize bag
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    # Set 1 for words present
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    
    return bag

# Example:
# all_words = ["cough", "fever", "headach", ...]
# sentence = ["I", "have", "coughing"]
# After stemming: ["i", "have", "cough"]
# bag = [1, 0, 0, ...]  (cough present, others absent)
```

**4. Flask Routes**

**Main Page:**
```python
@app.route('/')
def index():
    # Load symptoms for autocomplete
    data = []
    user_symptoms.clear()
    
    with open("static/assets/files/ds_symptoms.txt", "r") as file:
        all_symptoms = file.readlines()
    
    for s in all_symptoms:
        data.append(s.replace("'", "").replace("_", " ").replace(",\n", ""))
    
    data = json.dumps(data)
    return render_template('index.html', data=data)
```

**Symptom Processing:**
```python
@app.route('/symptom', methods=['POST'])
def predict_symptom():
    sentence = request.json['sentence']
    
    # Check if user finished entering symptoms
    if sentence.lower().strip().replace(".", "").replace("!", "") == "done":
        if not user_symptoms:
            response = "I can't know what disease you may have if you don't enter any symptoms :)"
        else:
            # Disease prediction
            disease = predict_disease(user_symptoms)
            response = format_disease_info(disease)
            user_symptoms.clear()
    else:
        # Extract symptom from sentence
        symptom, prob = get_symptom(sentence)
        
        if prob > 0.5:  # Confidence threshold
            response = f"Hmm, I'm {(prob * 100):.2f}% sure this is {symptom}."
            user_symptoms.add(symptom)
        else:
            response = "I'm sorry, but I don't understand you."
    
    return jsonify(response.replace("_", " "))
```

**5. Disease Prediction**
```python
def predict_disease(user_symptoms):
    # Create feature vector
    x_test = []
    for symptom in symptoms_list:
        if symptom in user_symptoms:
            x_test.append(1)
        else:
            x_test.append(0)
    
    x_test = np.asarray(x_test)
    
    # Predict
    disease = prediction_model.predict(x_test.reshape(1, -1))[0]
    
    return disease
```

**6. Information Retrieval**
```python
# Get disease description
description = diseases_description.loc[
    diseases_description['Disease'] == disease.lower().strip(), 
    'Description'
].iloc[0]

# Get precautions
precaution = disease_precaution[
    disease_precaution['Disease'] == disease.lower().strip()
]
precautions = (
    'Precautions: ' + 
    precaution.Precaution_1.iloc[0] + ", " + 
    precaution.Precaution_2.iloc[0] + ", " + 
    precaution.Precaution_3.iloc[0] + ", " + 
    precaution.Precaution_4.iloc[0]
)

# Format response
response = (
    f"It looks to me like you have {disease}. <br><br>" +
    f"<i>Description: {description}</i><br><br>" +
    f"<b>{precautions}</b>"
)
```

**7. Severity Assessment**
```python
# Calculate severity
severity = []
for symptom in user_symptoms:
    weight = symptom_severity.loc[
        symptom_severity['Symptom'] == symptom.lower().strip().replace(" ", ""), 
        'weight'
    ].iloc[0]
    severity.append(weight)

# Check if severe
if np.mean(severity) > 4 or np.max(severity) > 5:
    response += (
        "<br><br>Considering your symptoms are severe, " +
        "and Meddy isn't a real doctor, you should consider talking to one. :)"
    )
```

---

## 7. Coding Concepts

### Neural Network Architecture
```python
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out  # No softmax (handled by loss function)
```

### State Management (Session)
```python
user_symptoms = set()  # Global set to track symptoms per user

# Clear on new conversation
user_symptoms.clear()

# Add symptom
user_symptoms.add(symptom)
```

### Model Inference Mode
```python
model.eval()  # Disable dropout, batch norm training mode
with torch.no_grad():  # Disable gradient computation (faster)
    output = model(input)
```

### DataFrame Filtering
```python
# Efficient data retrieval
description = df.loc[df['Disease'] == disease, 'Description'].iloc[0]
```

---

## 8. Glossary

| Term | Definition |
|------|------------|
| **NLP** | Natural Language Processing |
| **Tokenization** | Splitting text into individual words/tokens |
| **Stemming** | Reducing words to root form (running → run) |
| **Bag of Words** | Text representation as word frequency vector |
| **Intent Classification** | Identifying user's intent from text |
| **PyTorch** | Deep learning framework |
| **Flask** | Python web framework |
| **ReLU** | Rectified Linear Unit activation function |
| **Softmax** | Converts logits to probabilities |
| **Cross-Entropy** | Loss function for classification |
| **Precaution** | Preventive measure for disease |
| **Severity** | Weight indicating symptom seriousness |
| **Forward Pass** | Computing output from input through network |
| **Inference Mode** | Model evaluation (no training) |

---

## 9. Outcomes & Results

### System Performance
- **Symptom Extraction Accuracy:** ~85-90% (with confidence > 0.5)
- **Disease Prediction:** Depends on symptom accuracy
- **Response Time:** <500ms per interaction

### Features
1. **Natural Language Understanding:** Understands varied phrasings
2. **Multi-Symptom Tracking:** Accumulates symptoms across conversation
3. **Confidence Scores:** Shows model certainty
4. **Medical Information:** Descriptions and precautions
5. **Severity Assessment:** Warns for severe symptoms

### Limitations
1. **Limited Symptom Vocabulary:** Only 132 symptoms
2. **No Context Memory:** Each sentence processed independently
3. **No Follow-up Questions:** Can't ask clarifying questions
4. **No Medical Reasoning:** Statistical patterns, not medical knowledge

---

## 10. Interview Questions & Answers

### Conceptual Questions

**Q1: Why use a two-stage architecture (NLP + ML) instead of end-to-end?**

**A1:** **Separation of Concerns:**

**Stage 1: NLP Model**
- **Task:** Extract structured symptom from unstructured text
- **Challenge:** Natural language is ambiguous, varied
- **Solution:** Neural network trained on symptom descriptions

**Stage 2: Disease Prediction**
- **Task:** Map symptoms to disease
- **Challenge:** Medical diagnosis logic
- **Solution:** Traditional ML (Decision Tree, Random Forest)

**Advantages:**
1. **Modularity:** Can improve each stage independently
2. **Interpretability:** Know which symptoms extracted
3. **Data Efficiency:** NLP needs text data, disease prediction needs symptom-disease mapping
4. **Flexibility:** Can add more diseases without retraining NLP

**End-to-End Alternative:**
```python
# Direct: Text → Disease
model = TransformerModel()
disease = model.predict("I have a headache and fever")
```
**Disadvantages:**
- Requires large labeled dataset (text, disease) pairs
- Less interpretable (black box)
- Can't accumulate symptoms across turns

**Q2: Explain the bag of words representation and its limitations.**

**A2:** **Bag of Words (BoW):**

**Representation:**
```python
Vocabulary: ["cough", "fever", "headache", ...]
Sentence: "I have a cough"
BoW: [1, 0, 0, ...]  # cough=1, others=0
```

**Advantages:**
1. **Simple:** Easy to implement and understand
2. **Fast:** Efficient computation
3. **Works Well:** Effective for intent classification

**Limitations:**

**1. No Word Order:**
```
"I have fever not cough" → Same as "I have cough not fever"
Both: [1, 1, 0, ...]
```

**2. No Semantics:**
```
"headache" and "migraine" are different words
BoW treats them as independent
No understanding they're related
```

**3. No Context:**
```
"I don't have a cough" → [1, ...] (cough=1)
Negative "don't" ignored!
```

**4. Sparse Representation:**
```
Vocabulary: 1000 words
Sentence: 5 words
Vector: 995 zeros, 5 ones (99.5% sparse)
```

**Better Alternatives:**
- **TF-IDF:** Weights words by importance
- **Word Embeddings:** Dense vectors capturing semantics
- **BERT:** Contextual representations
- **Attention Mechanisms:** Learn important words

**Q3: Why use PyTorch for NLP and Sklearn for disease prediction?**

**A3:**

**PyTorch for NLP:**
1. **Deep Learning:** Neural networks for complex pattern learning
2. **Text is Unstructured:** Need representation learning
3. **Flexibility:** Easy to customize architectures
4. **Dynamic Graphs:** Better for NLP (variable length sequences)

**Sklearn for Disease Prediction:**
1. **Structured Data:** Binary symptom vector (present/absent)
2. **Interpretability:** Tree-based models show decision rules
3. **Simplicity:** No need for deep learning on structured data
4. **Fast Training:** Decision trees train quickly
5. **Small Data:** Works well with limited symptom-disease pairs

**Example Disease Prediction:**
```python
# Input: [1, 0, 1, 0, 1, ...]  (fever=1, cough=0, headache=1, ...)
# Traditional ML sufficient

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_symptoms, y_diseases)

# No need for deep learning!
```

---

### Technical Questions

**Q4: Walk through the neural network forward pass for symptom extraction.**

**A4:**

**Example Input:** "I have a headache"

**Step 1: Tokenization**
```python
sentence = "I have a headache"
tokens = nltk.word_tokenize(sentence)
# tokens = ["I", "have", "a", "headache"]
```

**Step 2: Stemming**
```python
stemmed = [stemmer.stem(w.lower()) for w in tokens]
# stemmed = ["i", "have", "a", "headach"]
```

**Step 3: Bag of Words**
```python
# Vocabulary: ["i", "have", "a", "headach", "fever", "cough", ...]  (size: 1000)
bag = [1, 1, 1, 1, 0, 0, ...]  # First 4 words present
```

**Step 4: Convert to Tensor**
```python
X = torch.from_numpy(bag).reshape(1, 1000)  # (1, input_size)
```

**Step 5: Forward Pass**
```python
# Layer 1
h1 = self.l1(X)  # (1, 1000) → (1, 128)
h1 = self.relu(h1)  # Apply ReLU

# Layer 2
h2 = self.l2(h1)  # (1, 128) → (1, 128)
h2 = self.relu(h2)

# Layer 3
output = self.l3(h2)  # (1, 128) → (1, 132)  [132 symptoms]
```

**Step 6: Get Prediction**
```python
# Apply softmax
probs = torch.softmax(output, dim=1)
# probs = [0.001, 0.002, ..., 0.85, ..., 0.003]  (sum=1)

# Get symptom with highest probability
_, predicted = torch.max(probs, dim=1)
# predicted = 47 (index of "headache")

symptom = tags[47]  # "headache"
confidence = probs[0][47].item()  # 0.85 = 85%
```

**Q5: How does the confidence threshold (0.5) affect system behavior?**

**A5:**

**Current Implementation:**
```python
if prob > 0.5:
    response = f"I'm {prob*100:.2f}% sure this is {symptom}."
    user_symptoms.add(symptom)
else:
    response = "I'm sorry, but I don't understand you."
```

**Effect of Threshold:**

**Low Threshold (0.3):**
- **Pro:** Extracts more symptoms (high recall)
- **Con:** More false positives (user didn't mention)
- **Use Case:** Err on side of caution

**High Threshold (0.7):**
- **Pro:** Only confident predictions (high precision)
- **Con:** May miss some symptoms (low recall)
- **Use Case:** Avoid false alarms

**Optimal (0.5):**
- Balance between precision and recall
- Reasonable default

**Tuning Strategy:**
```python
# Adaptive threshold based on conversation state
if len(user_symptoms) == 0:
    threshold = 0.6  # Be confident for first symptom
else:
    threshold = 0.4  # More lenient for additional symptoms
```

**Q6: Explain the severity assessment. Why avg > 4 or max > 5?**

**A6:**

**Severity Weights:**
```csv
Symptom, weight
fever, 3
headache, 2
chest_pain, 7
difficulty_breathing, 8
mild_cough, 1
```

**Assessment Logic:**
```python
severity = []
for symptom in user_symptoms:
    weight = symptom_severity.loc[symptom, 'weight']
    severity.append(weight)

if np.mean(severity) > 4 or np.max(severity) > 5:
    # Alert: See a doctor!
```

**Conditions:**

**1. Average > 4:**
- Multiple moderate symptoms
- Example: [fever=3, headache=2, fatigue=3, nausea=3]
- Mean = 2.75 (no alert) vs [5, 4, 5, 4] → Mean = 4.5 (alert!)

**2. Maximum > 5:**
- At least one severe symptom
- Example: [chest_pain=7] → Max = 7 (alert!)
- Even if other symptoms mild

**Rationale:**
- **Conservative:** Better safe than sorry
- **Medical Priority:** Severe symptoms need immediate attention
- **Thresholds:** Based on domain expertise (could be tuned)

**Improvement:**
```python
# More nuanced assessment
def assess_severity(symptoms, weights):
    severity_scores = [weights[s] for s in symptoms]
    
    critical = any(w >= 8 for w in severity_scores)  # Critical symptom
    severe = np.max(severity_scores) >= 6  # Severe symptom
    moderate = np.mean(severity_scores) > 4  # Multiple moderate
    
    if critical:
        return "EMERGENCY: Seek immediate medical attention!"
    elif severe:
        return "See a doctor within 24 hours."
    elif moderate:
        return "Consider consulting a doctor."
    else:
        return "Monitor symptoms, rest, and hydrate."
```

---

### Implementation Questions

**Q7: How would you improve the chatbot to handle multi-turn conversations better?**

**A7:**

**Current Limitation:**
- No conversation memory (except symptoms)
- Can't ask follow-up questions
- No context awareness

**Improvements:**

**1. Conversation State Management**
```python
class ConversationState:
    def __init__(self):
        self.symptoms = set()
        self.history = []
        self.stage = "greeting"  # greeting, symptom_collection, diagnosis
        self.clarifications = {}
    
    def add_turn(self, user_input, bot_response):
        self.history.append({
            'user': user_input,
            'bot': bot_response,
            'timestamp': datetime.now()
        })
```

**2. Follow-up Questions**
```python
def ask_followup(symptom):
    followup_questions = {
        'headache': [
            "How long have you had the headache?",
            "On a scale of 1-10, how severe is the pain?",
            "Is it a throbbing or constant pain?"
        ],
        'fever': [
            "What's your temperature?",
            "How many days have you had the fever?",
            "Do you have chills?"
        ]
    }
    return random.choice(followup_questions.get(symptom, []))
```

**3. Context-Aware NLP**
```python
# Instead of processing sentence in isolation
def get_symptom_with_context(sentence, conversation_history):
    # Previous symptoms provide context
    context = " ".join([turn['user'] for turn in conversation_history[-3:]])
    full_text = context + " " + sentence
    
    # Extract symptom with context
    symptom, prob = nlp_model(full_text)
    return symptom, prob
```

**4. Intent Detection**
```python
class Intent(Enum):
    GREETING = "greeting"
    SYMPTOM = "symptom"
    QUESTION = "question"
    DONE = "done"
    CLARIFICATION = "clarification"

def detect_intent(sentence):
    if sentence.lower() in ["hi", "hello", "hey"]:
        return Intent.GREETING
    elif "?" in sentence:
        return Intent.QUESTION
    elif sentence.lower() in ["done", "finished", "that's all"]:
        return Intent.DONE
    else:
        return Intent.SYMPTOM
```

**5. Dialogue Management**
```python
def manage_dialogue(user_input, state):
    intent = detect_intent(user_input)
    
    if intent == Intent.GREETING:
        state.stage = "symptom_collection"
        return "Hello! Tell me your symptoms."
    
    elif intent == Intent.SYMPTOM:
        symptom, prob = get_symptom(user_input)
        state.symptoms.add(symptom)
        
        # Ask follow-up
        followup = ask_followup(symptom)
        return f"I see you have {symptom}. {followup}"
    
    elif intent == Intent.DONE:
        disease = predict_disease(state.symptoms)
        return format_disease_info(disease)
```

**Q8: How would you deploy this chatbot for production use?**

**A8:**

**Production Considerations:**

**1. Scalability**
```python
# Use gunicorn for multiple workers
gunicorn --workers 4 --bind 0.0.0.0:5000 app:app

# Load model once, share across workers
# Move model loading outside Flask app
```

**2. Session Management**
```python
# Instead of global user_symptoms, use sessions
from flask import session

@app.route('/symptom', methods=['POST'])
def predict_symptom():
    if 'symptoms' not in session:
        session['symptoms'] = []
    
    symptoms = session['symptoms']
    # ... process ...
    session['symptoms'] = symptoms
```

**3. Database Integration**
```python
# Store conversations for analysis
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    symptoms = Column(String)  # JSON
    predicted_disease = Column(String)
    timestamp = Column(DateTime)

# Log each conversation
db.session.add(Conversation(user_id=user_id, symptoms=json.dumps(symptoms)))
db.session.commit()
```

**4. API Rate Limiting**
```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/symptom', methods=['POST'])
@limiter.limit("10 per minute")  # Prevent abuse
def predict_symptom():
    ...
```

**5. HTTPS and Security**
```python
# Use HTTPS for medical data
# Encrypt conversations
# Implement user authentication
from flask_login import login_required

@app.route('/symptom', methods=['POST'])
@login_required
def predict_symptom():
    ...
```

**6. Monitoring and Logging**
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/symptom', methods=['POST'])
def predict_symptom():
    try:
        # ... process ...
        logger.info(f"Predicted symptom: {symptom}, confidence: {prob}")
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": "Internal server error"}), 500
```

**7. Model Updates**
```python
# Mechanism to update models without downtime
# Use model versioning
MODEL_VERSION = "v2.0"

def load_model(version):
    model_path = f"models/data_{version}.pth"
    return torch.load(model_path)

# Hot reload on new version
@app.route('/update_model', methods=['POST'])
@admin_required
def update_model():
    global nlp_model
    nlp_model = load_model(new_version)
    return jsonify({"status": "Model updated"})
```

---

## Additional Resources

**Papers:**
- Wei et al. (2018): "Task-Oriented Dialogue System for Automatic Diagnosis"
- Xu et al. (2019): "End-to-End Knowledge-Rooted Dialogue System for Automatic Diagnosis"

**Datasets:**
- Medical Dialog Dataset
- MedQuAD (Medical Question Answering Dataset)
- HealthTap Dataset

**Frameworks:**
- Rasa: Open-source conversational AI
- Dialogflow: Google's chatbot platform
- Microsoft Bot Framework

