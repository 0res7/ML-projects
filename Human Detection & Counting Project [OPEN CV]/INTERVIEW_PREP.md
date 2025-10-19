# Interview Preparation: Human Detection & Counting Project

## 1. Project Overview

**Problem Statement:** Develop a system to detect and count people in images, videos, or live webcam feeds using Histogram of Oriented Gradients (HOG) descriptor with SVM classifier.

**Objective:** Build a real-time human detection and counting application for applications like crowd monitoring, retail analytics, security systems, and social distancing enforcement.

**Use Cases:**
- Occupancy monitoring in buildings
- Crowd management at events
- Retail customer counting
- Social distancing compliance
- Security surveillance

---

## 2. Technical Concepts

### HOG (Histogram of Oriented Gradients)
- **Feature Descriptor:** Captures edge direction and magnitude
- **SVM Classifier:** Linear Support Vector Machine for classification
- **Sliding Window:** Scans image at multiple scales
- **Non-Maximum Suppression:** Removes overlapping detections

### HOG Pipeline
1. **Gradient Computation:** Calculate gradient magnitude and direction
2. **Cell Histograms:** Divide image into cells, create orientation histograms
3. **Block Normalization:** Normalize histograms across blocks
4. **Feature Vector:** Concatenate all block histograms
5. **SVM Classification:** Classify feature vector as human/non-human

---

## 3. Libraries & Technologies

### Core Libraries
- **OpenCV (cv2):** HOG detector, image processing
- **imutils:** Image resizing utilities
- **NumPy:** Array operations
- **argparse:** Command-line arguments

### HOG Detector
```python
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
```

---

## 4. Code Architecture & Design Patterns

### File Structure
```
Human Detection & Counting Project [OPEN CV]/
├── human-counting-project-code.py    # Main script
├── images.jpg                         # Sample image
└── README.md
```

### Design Pattern: Strategy Pattern
```python
# Different detection strategies
def detectByPathImage(path, output_path):
    # Strategy for single image
    
def detectByPathVideo(path, writer):
    # Strategy for video file
    
def detectByCamera(writer):
    # Strategy for webcam
```

### Main Detection Function
```python
def detect(frame):
    # HOG detection
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(
        frame,
        winStride=(4, 4),
        padding=(8, 8),
        scale=1.03
    )
    
    person = 1
    for x, y, w, h in bounding_box_cordinates:
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'person {person}', (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        person += 1
    
    # Display total count
    cv2.putText(frame, f'Total Persons : {person-1}', (40, 70),
               cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    
    return frame
```

---

## 5. Mathematical Foundations

### HOG Feature Extraction

**Step 1: Gradient Computation**
\[
G_x = I * \begin{bmatrix} -1 & 0 & 1 \end{bmatrix}, \quad G_y = I * \begin{bmatrix} -1 \\ 0 \\ 1 \end{bmatrix}
\]

**Magnitude and Orientation:**
\[
|G| = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\left(\frac{G_y}{G_x}\right)
\]

**Step 2: Cell Histograms**
- Divide image into 8×8 pixel cells
- Create 9-bin histogram of gradient orientations (0°-180°)
- Weight each pixel by gradient magnitude

**Step 3: Block Normalization**
- Group cells into 2×2 cell blocks
- Normalize histogram across block:
\[
v_{\text{norm}} = \frac{v}{\sqrt{||v||_2^2 + \epsilon^2}}
\]

**Step 4: Feature Vector**
- Concatenate all normalized block histograms
- Typical size: 64×128 image → 3,780-dimensional feature vector

### SVM Decision Function
\[
f(x) = w^T x + b
\]
- If \(f(x) > 0\): Human detected
- If \(f(x) \leq 0\): Background

### detectMultiScale Parameters

**winStride:** Step size for sliding window
- Smaller → More detections, slower
- Larger → Fewer detections, faster

**padding:** Pixels added around detection window
- Provides context for better detection

**scale:** Scale factor for image pyramid
- 1.03 = 3% increase per level
- Balances speed vs accuracy

---

## 6. Implementation Details

### Complete Script Breakdown

**1. Initialization**
```python
import cv2
import imutils
import numpy as np
import argparse

def detect(frame):
    # Detect humans
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(
        frame, 
        winStride=(4, 4), 
        padding=(8, 8), 
        scale=1.03
    )
    
    person = 1
    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f'person {person}', (x,y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        person += 1
    
    cv2.putText(frame, 'Status : Detecting ', (40,40), 
               cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.putText(frame, f'Total Persons : {person-1}', (40,70), 
               cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.imshow('output', frame)
    
    return frame
```

**2. Image Detection**
```python
def detectByPathImage(path, output_path):
    image = cv2.imread(path)
    image = imutils.resize(image, width=min(800, image.shape[1]))
    result_image = detect(image)
    
    if output_path is not None:
        cv2.imwrite(output_path, result_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

**3. Video Detection**
```python
def detectByPathVideo(path, writer):
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    
    if check == False:
        print('Video Not Found. Please Enter a Valid Path.')
        return
    
    print('Detecting people...')
    while video.isOpened():
        check, frame = video.read()
        
        if check:
            frame = imutils.resize(frame, width=min(800, frame.shape[1]))
            frame = detect(frame)
            
            if writer is not None:
                writer.write(frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    
    video.release()
    cv2.destroyAllWindows()
```

**4. Webcam Detection**
```python
def detectByCamera(writer):   
    video = cv2.VideoCapture(0)
    print('Detecting people...')
    
    while True:
        check, frame = video.read()
        frame = detect(frame)
        
        if writer is not None:
            writer.write(frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()
```

**5. Main Function**
```python
if __name__ == "__main__":
    # Initialize HOG detector
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Parse arguments
    args = argsParser()
    humanDetector(args)
```

---

## 7. Coding Concepts

### Command-Line Interface
```python
def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File")
    arg_parse.add_argument("-i", "--image", default=None, help="path to Image File")
    arg_parse.add_argument("-c", "--camera", default=False, help="Set true for camera")
    arg_parse.add_argument("-o", "--output", type=str, help="path to optional output")
    args = vars(arg_parse.parse_args())
    return args
```

**Usage:**
```bash
# Detect in image
python human-counting-project-code.py -i images.jpg -o output.jpg

# Detect in video
python human-counting-project-code.py -v video.mp4 -o output.avi

# Detect from webcam
python human-counting-project-code.py -c true -o webcam.avi
```

### Video Writer
```python
writer = cv2.VideoWriter(
    args['output'],
    cv2.VideoWriter_fourcc(*'MJPG'),
    10,  # FPS
    (600, 600)  # Frame size
)
```

### Image Resizing
```python
frame = imutils.resize(frame, width=min(800, frame.shape[1]))
```
- Limits width to 800 pixels (maintains aspect ratio)
- Faster processing on smaller images

---

## 8. Glossary

| Term | Definition |
|------|------------|
| **HOG** | Histogram of Oriented Gradients feature descriptor |
| **SVM** | Support Vector Machine classifier |
| **Gradient** | Rate of change in pixel intensity |
| **Orientation** | Direction of gradient (angle) |
| **Cell** | Small region (e.g., 8×8 pixels) for histogram computation |
| **Block** | Group of cells (e.g., 2×2 cells) for normalization |
| **Sliding Window** | Technique to scan image at multiple locations |
| **Image Pyramid** | Multi-scale representation of image |
| **winStride** | Step size for sliding window |
| **Non-Maximum Suppression** | Remove overlapping detections |
| **Bounding Box** | Rectangle enclosing detected person |

---

## 9. Outcomes & Results

### Performance Metrics
- **Detection Accuracy:** ~85-90% in ideal conditions
- **False Positives:** ~10-15% (especially with partial occlusions)
- **Processing Speed:** 5-15 FPS (depends on image size and scale parameter)

### Limitations
1. **Occlusion:** Struggles with partially hidden people
2. **Crowded Scenes:** Overlapping detections
3. **Pose Variations:** Works best with upright, frontal people
4. **Scale:** May miss very small or very large people

---

## 10. Interview Questions & Answers

### Conceptual Questions

**Q1: What is HOG (Histogram of Oriented Gradients) and how does it work?**

**A1:** HOG is a feature descriptor that captures edge directions and magnitudes in an image.

**Process:**
1. **Compute Gradients:** Calculate horizontal and vertical gradients
2. **Create Cell Histograms:** Divide image into cells (8×8), bin gradient orientations (9 bins)
3. **Normalize Blocks:** Group cells into blocks (2×2), normalize histograms
4. **Feature Vector:** Concatenate all normalized histograms

**Why HOG for Human Detection:**
- Humans have characteristic edge patterns (head, shoulders, legs)
- Robust to lighting changes
- Translation invariant (within cell)

**Q2: Explain the sliding window and image pyramid approach.**

**A2:** 

**Sliding Window:**
- Scan fixed-size window across image
- At each position, extract HOG features and classify
- winStride=(4, 4): Move window 4 pixels at a time

**Image Pyramid:**
- Create multiple scaled versions of image
- scale=1.03: Each level 3% larger
- Detects people at different sizes

**Example:**
```
Level 0: 640×480 (original)
Level 1: 659×494 (1.03×)
Level 2: 679×509 (1.03²×)
...
```

### Technical Questions

**Q3: How does detectMultiScale work? Explain the parameters.**

**A3:**

```python
bounding_box_cordinates, weights = HOGCV.detectMultiScale(
    frame,
    winStride=(4, 4),    # Sliding window step
    padding=(8, 8),      # Pixels added around window
    scale=1.03           # Image pyramid scale factor
)
```

**Parameters:**

**winStride=(4, 4):**
- Horizontal and vertical step size
- Smaller = more detections, slower
- Larger = fewer detections, faster

**padding=(8, 8):**
- Extra pixels around detection window
- Provides context for better classification
- Helps detect people near edges

**scale=1.03:**
- Factor for image pyramid
- 1.03 = 3% size increase per level
- Smaller = more scales, more accurate, slower
- Larger = fewer scales, faster, may miss detections

**Returns:**
- **bounding_box_cordinates:** List of [x, y, w, h] for each detection
- **weights:** Confidence scores (SVM decision function values)

**Q4: What are the limitations of HOG-based human detection?**

**A4:**

**1. Occlusion:**
- Problem: Partially hidden people not detected
- HOG expects full silhouette
- Solution: Part-based models, deep learning

**2. Crowded Scenes:**
- Problem: Overlapping people confused
- Multiple detections for single person
- Solution: Non-Maximum Suppression, tracking

**3. Pose Variations:**
- Problem: Trained on upright, frontal poses
- Fails on sitting, lying down, back views
- Solution: Train on diverse poses, deformable parts

**4. Computational Cost:**
- Problem: Sliding window + image pyramid = slow
- O(scales × windows × feature_extraction)
- Solution: Use deep learning (YOLO, SSD) for speed

**5. Illumination:**
- HOG partially robust but extreme lighting affects gradients
- Solution: Preprocessing (histogram equalization)

**Q5: How would you implement Non-Maximum Suppression to remove duplicate detections?**

**A5:**

```python
def non_max_suppression(boxes, weights, overlap_thresh=0.3):
    """
    Remove overlapping bounding boxes.
    
    Args:
        boxes: List of [x, y, w, h]
        weights: Confidence scores
        overlap_thresh: IoU threshold (0.3 = 30% overlap)
    
    Returns:
        Filtered boxes and weights
    """
    if len(boxes) == 0:
        return [], []
    
    # Convert to (x1, y1, x2, y2)
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    # Sort by confidence (weights)
    weights = np.array(weights)
    idxs = np.argsort(weights)
    
    pick = []
    
    while len(idxs) > 0:
        # Pick box with highest confidence
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # Find overlap with remaining boxes
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # Compute IoU
        intersection = w * h
        union = (x2[i] - x1[i] + 1) * (y1[i] - y1[i] + 1) + \
                (x2[idxs[:last]] - x1[idxs[:last]] + 1) * \
                (y2[idxs[:last]] - y1[idxs[:last]] + 1) - intersection
        overlap = intersection / union
        
        # Remove boxes with high overlap
        idxs = np.delete(idxs, np.concatenate(([last], 
                        np.where(overlap > overlap_thresh)[0])))
    
    return boxes[pick], weights[pick]

# Usage
boxes, weights = HOGCV.detectMultiScale(frame, ...)
boxes_nms, weights_nms = non_max_suppression(boxes, weights)
```

### Implementation Questions

**Q6: How would you optimize this for real-time performance?**

**A6:**

**1. Reduce Image Resolution:**
```python
# Resize to smaller width
frame = imutils.resize(frame, width=400)  # Instead of 800
```

**2. Increase winStride:**
```python
# Larger steps = fewer windows
bounding_box_cordinates, weights = HOGCV.detectMultiScale(
    frame,
    winStride=(8, 8),  # Instead of (4, 4)
    padding=(8, 8),
    scale=1.05  # Larger scale = fewer pyramid levels
)
```

**3. ROI Processing:**
```python
# Only process region of interest
roi = frame[100:500, 200:600]  # Crop to relevant area
detections = HOGCV.detectMultiScale(roi, ...)
```

**4. Frame Skipping:**
```python
frame_count = 0
cached_detections = []

while True:
    ret, frame = video.read()
    
    if frame_count % 3 == 0:  # Process every 3rd frame
        cached_detections = HOGCV.detectMultiScale(frame, ...)
    else:
        # Use cached detections
        for (x, y, w, h) in cached_detections:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    frame_count += 1
```

**5. Use Deep Learning (Faster):**
```python
# YOLO v3/v4 for real-time detection (30+ FPS)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
```

**Q7: Implement crowd density estimation based on detection count.**

**A7:**

```python
class CrowdMonitor:
    def __init__(self, frame_width, frame_height):
        self.frame_area = frame_width * frame_height
        self.history = []
        self.window_size = 30  # 1 second at 30 FPS
    
    def update(self, person_count):
        self.history.append(person_count)
        if len(self.history) > self.window_size:
            self.history.pop(0)
    
    def get_average_count(self):
        return np.mean(self.history) if self.history else 0
    
    def get_density(self):
        """People per square meter (assuming frame is 10m²)."""
        avg_count = self.get_average_count()
        return avg_count / 10.0  # Adjust based on actual coverage
    
    def get_crowd_level(self):
        """Classify crowd level."""
        density = self.get_density()
        if density < 0.5:
            return "LOW"
        elif density < 2.0:
            return "MEDIUM"
        elif density < 4.0:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def is_social_distancing_violated(self, bounding_boxes):
        """Check if people too close (simplified)."""
        violations = 0
        for i, box1 in enumerate(bounding_boxes):
            for box2 in bounding_boxes[i+1:]:
                # Compute distance between centers
                center1 = (box1[0] + box1[2]/2, box1[1] + box1[3]/2)
                center2 = (box2[0] + box2[2]/2, box2[1] + box2[3]/2)
                distance = np.sqrt(
                    (center1[0] - center2[0])**2 + 
                    (center1[1] - center2[1])**2
                )
                # Violation if distance < 50 pixels (adjust based on scale)
                if distance < 50:
                    violations += 1
        return violations

# Usage
monitor = CrowdMonitor(800, 600)

while True:
    ret, frame = video.read()
    bounding_boxes, weights = HOGCV.detectMultiScale(frame, ...)
    
    person_count = len(bounding_boxes)
    monitor.update(person_count)
    
    # Display statistics
    cv2.putText(frame, f'Count: {person_count}', (10, 30), ...)
    cv2.putText(frame, f'Avg: {monitor.get_average_count():.1f}', (10, 60), ...)
    cv2.putText(frame, f'Level: {monitor.get_crowd_level()}', (10, 90), ...)
    
    violations = monitor.is_social_distancing_violated(bounding_boxes)
    cv2.putText(frame, f'SD Violations: {violations}', (10, 120), ...)
```

---

## Additional Resources

**Papers:**
- Dalal & Triggs (2005): "Histograms of Oriented Gradients for Human Detection"
- Felzenszwalb et al. (2010): "Object Detection with Discriminatively Trained Part-Based Models"

**Alternative Methods:**
- **YOLO:** Real-time object detection (faster)
- **Mask R-CNN:** Instance segmentation (more accurate)
- **OpenPose:** Keypoint detection for pose estimation

