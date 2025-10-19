# Interview Preparation: Human Activity Detection

## 1. Project Overview

**Problem Statement:** Recognize human activities (walking, running, sitting, standing, etc.) from video sequences using deep learning models for action recognition.

**Objective:** Build a video classification system using LSTM (Long Short-Term Memory) networks and/or Detectron2 for human pose estimation and action recognition.

**Applications:**
- Surveillance and security
- Healthcare monitoring (fall detection, activity tracking)
- Sports analytics  
- Smart home automation
- Human-computer interaction

---

## 2. Technical Concepts

### Video Classification
- **Temporal Modeling:** Actions unfold over time (sequences of frames)
- **Spatial Features:** What's in each frame (people, objects)
- **Spatio-Temporal:** Combining space and time

### Approaches
1. **LSTM on Frame Features:** Extract features per frame → LSTM
2. **3D CNN:** Convolution across spatial and temporal dimensions
3. **Two-Stream Networks:** RGB + Optical Flow
4. **Pose-Based:** Human keypoints → Action classification

### Models
- **LSTM:** Recurrent neural network for sequences
- **Detectron2:** Facebook's object detection and pose estimation
- **CNN + LSTM:** Hybrid architecture

---

## 3. Mathematical Foundations

### LSTM Cell
\[
\begin{align}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(input gate)} \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{(candidate)} \\
C_t &= f_t \times C_{t-1} + i_t \times \tilde{C}_t \quad \text{(cell state)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(output gate)} \\
h_t &= o_t \times \tanh(C_t) \quad \text{(hidden state)}
\end{align}
\]

### Categorical Cross-Entropy (Multi-Class)
\[
L = -\sum_{i=1}^{N}\sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})
\]

### Optical Flow
Motion between frames:
\[
I(x, y, t) = I(x + \Delta x, y + \Delta y, t + \Delta t)
\]

---

## 4. Implementation Details

### Architecture: CNN + LSTM
```python
import tensorflow as tf
from tensorflow import keras
from keras.applications import ResNet50
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, GlobalAveragePooling2D
from keras.models import Sequential, Model

# Feature extraction: Pre-trained CNN
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze CNN

# Add pooling
x = GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(inputs=base_model.input, outputs=x)

# Full model: Sequence of frames → LSTM
sequence_input = keras.Input(shape=(sequence_length, 224, 224, 3))
x = TimeDistributed(feature_extractor)(sequence_input)  # Apply CNN to each frame
x = LSTM(128, return_sequences=True)(x)
x = Dropout(0.5)(x)
x = LSTM(64)(x)
x = Dropout(0.5)(x)
output = Dense(num_actions, activation='softmax')(x)

model = Model(inputs=sequence_input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Data Preparation
```python
import cv2

def extract_frames(video_path, num_frames=30):
    """Extract fixed number of frames from video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames uniformly
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0  # Normalize
            frames.append(frame)
    
    cap.release()
    return np.array(frames)

# Create dataset
X = []
y = []

for video_file, label in dataset:
    frames = extract_frames(video_file)
    X.append(frames)
    y.append(label)

X = np.array(X)  # Shape: (num_videos, num_frames, 224, 224, 3)
y = keras.utils.to_categorical(y, num_classes=num_actions)
```

### Training
```python
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Callbacks
callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5)
]

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=8,  # Small batch due to memory
    callbacks=callbacks
)

# Plot training history
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
```

### Inference (Real-Time)
```python
def predict_action(video_path):
    frames = extract_frames(video_path, num_frames=30)
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    
    predictions = model.predict(frames)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    action = action_labels[predicted_class]
    return action, confidence

# Flask app
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    video_file = request.files['video']
    video_path = save_temp(video_file)
    
    action, confidence = predict_action(video_path)
    
    return jsonify({
        'action': action,
        'confidence': float(confidence)
    })
```

---

## 5. Outcomes & Results

### Typical Performance
- **Accuracy:** 75-90% (depends on dataset complexity)
- **Inference Speed:** 0.5-2 seconds per video (CPU)
- **Common Actions:** Walking, running, sitting, standing, waving

### Challenges
1. **Viewpoint Variation:** Different camera angles
2. **Occlusion:** Partially hidden people
3. **Background Clutter:** Complex scenes
4. **Similar Actions:** Walking vs jogging hard to distinguish

---

## 6. Interview Questions & Answers

**Q1: Why use LSTM for video classification?**

**A1:**

**Sequential Nature of Video:**
- Frame 1 → Frame 2 → ... → Frame N
- Action spans multiple frames (temporal dependencies)
- LSTM remembers past frames

**vs CNN:**
- CNN treats frames independently
- LSTM models temporal dynamics
- Better for motion-based actions

**Q2: What is the difference between 2D CNN and 3D CNN for videos?**

**A2:**

**2D CNN:**
- Convolution across (height, width)
- Applied per frame
- Spatial features only

**3D CNN:**
- Convolution across (height, width, time)
- Captures motion directly
- Spatio-temporal features

**Trade-offs:**
- 3D CNN: Better accuracy, more parameters, slower
- 2D CNN + LSTM: Good balance

**Q3: How would you improve activity recognition accuracy?**

**A3:**

**1. Optical Flow:**
```python
# Add motion information
flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, ...)
# Two-stream network: RGB + Flow
```

**2. Skeleton/Pose:**
```python
# Use Detectron2 for keypoints
from detectron2.utils.visualizer import Visualizer
# Extract 17 keypoints per person
# Classify based on pose sequences
```

**3. Temporal Augmentation:**
```python
# Vary video speed, crop temporal segments
augmentations = [
    ('speed_up', 1.5),
    ('slow_down', 0.75),
    ('temporal_crop', (0.1, 0.9))
]
```

**4. Transfer Learning:**
```python
# Pre-trained on large action dataset (Kinetics-400)
base = load_pretrained_model('kinetics400')
# Fine-tune on specific domain
```

**5. Ensemble:**
```python
# Combine multiple models
predictions = 0.4 * lstm_pred + 0.3 * 3dcnn_pred + 0.3 * pose_pred
```

---

## Additional Resources

**Datasets:**
- UCF101: 101 action categories
- Kinetics-400/700: Large-scale video dataset
- HMDB51: Human motion database

**Models:**
- I3D (Inflated 3D ConvNet)
- SlowFast Networks
- TimeSformer (Transformer for video)

