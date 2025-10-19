# Interview Preparation: Colorize Black & White Images

## 1. Project Overview

**Problem Statement:** Automatically colorize grayscale images using deep learning, transforming black and white photos into realistic color images.

**Objective:** Implement an end-to-end image colorization system using a pre-trained Caffe model that predicts color channels (a, b) in LAB color space from grayscale (L channel) input.

**Applications:**
- Restore old family photographs
- Colorize historical images
- Film restoration
- Artistic colorization

---

## 2. Technical Concepts

### Color Spaces
- **RGB:** Red-Green-Blue (additive color model)
- **LAB:** L (Lightness), a (green-red), b (blue-yellow)
  - L: 0-100 (black to white)
  - a: -128 to +127 (green to red)
  - b: -128 to +127 (blue to yellow)

### Why LAB for Colorization?
1. **Separates Luminance from Color:** L channel is independent of chrominance
2. **Perceptually Uniform:** Changes in values correspond to perceived color differences
3. **Efficient:** Only predict a and b channels, L comes from grayscale

### Deep Learning Architecture
- **Encoder-Decoder CNN:** Bottleneck architecture
- **Class Rebalancing:** Handle class imbalance in color distribution
- **Pre-trained Model:** Trained on ImageNet (1M+ images)

---

## 3. Libraries & Technologies

### Core Libraries
- **OpenCV (cv2):** Image I/O, preprocessing, color space conversion
  - `cv2.imread()`: Read images
  - `cv2.cvtColor()`: Convert between color spaces
  - `cv2.dnn`: Deep neural network inference
  - `cv2.resize()`: Resize images
  - `cv2.imwrite()`: Save results

- **NumPy:** Array operations, tensor manipulation
- **Matplotlib:** Visualization (optional)

### Model Files
```
colorization_deploy_v2.prototxt     # Model architecture (Caffe)
colorization_release_v2.caffemodel  # Trained weights
pts_in_hull.npy                     # Quantized ab color space points
```

---

## 4. Code Architecture & Design Patterns

### File Structure
```
Colorize Black & white images [OPEN CV]/
├── image_colarization.py           # Main script
├── models/
│   ├── colorization_deploy_v2.prototxt
│   └── colorization_release_v2.caffemodel
├── pts_in_hull.npy                 # 313 color cluster centers
├── new.jpg                         # Input grayscale image
└── result.png                      # Output colorized image
```

### Processing Pipeline
```
Grayscale Image → Convert to LAB → Extract L channel → 
Resize → Mean centering → DNN Inference → 
Predict ab channels → Resize ab → Combine Lab → 
Convert to BGR → Save
```

### Design Pattern: Pipeline
```python
# Sequential processing steps
frame = load_image()
lab_img = convert_to_lab(frame)
l_channel = extract_l_channel(lab_img)
l_resized = resize(l_channel, (224, 224))
l_normalized = normalize(l_resized)
ab_predicted = model.forward(l_normalized)
ab_resized = resize_ab(ab_predicted)
colorized = combine_and_convert(l_channel, ab_resized)
save_image(colorized)
```

---

## 5. Mathematical Foundations

### LAB Color Space Conversion

**RGB to LAB:**
1. RGB → XYZ (linear transformation)
\[
\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = M \times \begin{bmatrix} R \\ G \\ B \end{bmatrix}
\]

2. XYZ → LAB (non-linear transformation)
\[
L = 116 \times f(Y/Y_n) - 16
\]
\[
a = 500 \times (f(X/X_n) - f(Y/Y_n))
\]
\[
b = 200 \times (f(Y/Y_n) - f(Z/Z_n))
\]

where \(f(t) = \begin{cases} t^{1/3} & \text{if } t > \delta^3 \\ \frac{t}{3\delta^2} + \frac{4}{29} & \text{otherwise} \end{cases}\)

### Mean Centering (L Channel)
\[
L_{\text{normalized}} = L - 50
\]
Centers L channel around zero (L ranges 0-100, centered at 50).

### Quantized ab Space
- Continuous ab space quantized to 313 bins
- Each bin represents common color in natural images
- Reduces output dimensionality: Continuous → 313 classes

### Softmax (Implicit)
Model outputs 313 probabilities:
\[
P(bin_i | L) = \frac{e^{z_i}}{\sum_{j=1}^{313} e^{z_j}}
\]

### Weighted Sum for Final ab
\[
\begin{bmatrix} a \\ b \end{bmatrix} = \sum_{i=1}^{313} P(bin_i | L) \times \begin{bmatrix} a_i \\ b_i \end{bmatrix}
\]

---

## 6. Implementation Details

### Step-by-Step Code Walkthrough

**1. Load Image and Model**
```python
# Read grayscale/color image
frame = cv.imread("new.jpg")

# Load quantized ab values (313 color bins)
numpy_file = np.load('./pts_in_hull.npy')  # Shape: (313, 2)

# Load Caffe model
Caffe_net = cv.dnn.readNetFromCaffe(
    "./models/colorization_deploy_v2.prototxt",
    "./models/colorization_release_v2.caffemodel"
)
```

**2. Add Color Centers to Model**
```python
# Reshape to (2, 313, 1, 1) for Conv layer
numpy_file = numpy_file.transpose().reshape(2, 313, 1, 1)

# Set as weights for class8_ab layer
Caffe_net.getLayer(Caffe_net.getLayerId('class8_ab')).blobs = [
    numpy_file.astype(np.float32)
]

# Set scaling factor for conv8_313_rh layer
Caffe_net.getLayer(Caffe_net.getLayerId('conv8_313_rh')).blobs = [
    np.full([1, 313], 2.606, np.float32)
]
```

**3. Preprocess Image**
```python
# Convert BGR to RGB, normalize to [0, 1]
rgb_img = (frame[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)

# Convert RGB to LAB
lab_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2Lab)

# Extract L channel
l_channel = lab_img[:,:,0]  # Shape: (H, W)

# Resize to network input size
input_width, input_height = 224, 224
l_channel_resize = cv.resize(l_channel, (input_width, input_height))

# Mean center (subtract 50)
l_channel_resize -= 50
```

**4. Inference**
```python
# Create blob (add batch and channel dimensions)
Caffe_net.setInput(cv.dnn.blobFromImage(l_channel_resize))

# Forward pass
ab_channel = Caffe_net.forward()[0,:,:,:].transpose((1,2,0))
# Shape: (56, 56, 2) → a and b channels
```

**5. Postprocess and Combine**
```python
# Get original dimensions
(original_height, original_width) = rgb_img.shape[:2]

# Resize ab channels to original size
ab_channel_us = cv.resize(ab_channel, (original_width, original_height))

# Combine L with predicted ab
lab_output = np.concatenate(
    (l_channel[:,:,np.newaxis], ab_channel_us), 
    axis=2
)

# Convert LAB to BGR
bgr_output = np.clip(cv.cvtColor(lab_output, cv.COLOR_Lab2BGR), 0, 1)

# Save result
cv.imwrite("./result.png", (bgr_output*255).astype(np.uint8))
```

---

## 7. Coding Concepts

### Array Manipulation
```python
# Transpose: (313, 2) → (2, 313)
numpy_file.transpose()

# Reshape: (2, 313) → (2, 313, 1, 1)
.reshape(2, 313, 1, 1)

# Add dimension: (H, W) → (H, W, 1)
l_channel[:,:,np.newaxis]

# Concatenate along axis 2 (channels)
np.concatenate((l_channel, ab_channel), axis=2)
```

### Color Channel Indexing
```python
# BGR to RGB: Reverse channel order
frame[:,:,[2, 1, 0]]  # [B, G, R] → [R, G, B]

# Extract single channel
l_channel = lab_img[:,:,0]  # First channel (L)
```

### Type Conversion
```python
# Int to float
(frame * 1.0 / 255).astype(np.float32)

# Float to uint8
(bgr_output * 255).astype(np.uint8)
```

### Clipping
```python
# Ensure values in valid range [0, 1]
np.clip(bgr_output, 0, 1)
```

---

## 8. Glossary

| Term | Definition |
|------|------------|
| **LAB Color Space** | Color space separating lightness (L) from color (a, b) |
| **L Channel** | Lightness channel (0-100), represents grayscale |
| **a Channel** | Green (-128) to Red (+127) color axis |
| **b Channel** | Blue (-128) to Yellow (+127) color axis |
| **Colorization** | Process of adding color to grayscale images |
| **Caffe** | Deep learning framework by Berkeley AI Research |
| **Blob** | 4D tensor input for neural networks [N, C, H, W] |
| **pts_in_hull** | Quantized ab color space (313 representative colors) |
| **Class Rebalancing** | Technique to handle rare colors in training |
| **Encoder-Decoder** | Architecture that compresses then reconstructs data |
| **Mean Centering** | Subtracting mean to center data around zero |
| **Upsampling** | Increasing spatial resolution (opposite of downsampling) |

---

## 9. Outcomes & Results

### Model Specifications
- **Architecture:** Encoder-decoder CNN with skip connections
- **Input Size:** 224×224×1 (L channel)
- **Output Size:** 56×56×2 (ab channels, upsampled to original size)
- **Training Data:** ImageNet (1.3M images)
- **Quantization:** 313 color bins in ab space

### Performance
- **Speed:** ~1-3 seconds per image (CPU)
- **Quality:** Realistic colors for common objects (sky, grass, skin)
- **Limitations:** May produce desaturated/incorrect colors for ambiguous scenes

---

## 10. Interview Questions & Answers

### Conceptual Questions

**Q1: Why use LAB color space instead of RGB for colorization?**

**A1:** LAB color space is superior for colorization for several reasons:

**1. Separates Luminance from Chrominance:**
- L channel (lightness) is independent of color
- Grayscale image directly provides L channel
- Only need to predict a and b channels

**2. Perceptually Uniform:**
- Equal distances in LAB space correspond to equal perceptual differences
- Easier for model to learn meaningful color relationships

**3. Smaller Color Space:**
- RGB: 256^3 = 16.7M possible colors
- LAB (quantized): 313 common colors
- Reduces complexity, focuses on natural colors

**4. Mathematical Convenience:**
\[
\text{Grayscale} \xrightarrow{1:1} L \text{ channel}
\]
\[
\text{Colorization} = \text{Predict } (a, b) \text{ given } L
\]

**In RGB:**
- All three channels interdependent
- Must predict 3 values simultaneously
- More complex problem

**Q2: Explain the concept of "quantized ab space" with 313 bins.**

**A2:** Quantized ab space reduces continuous color space to discrete bins.

**Continuous vs Quantized:**
- **Continuous:** a and b can be any value in [-128, 127]
  - Infinite possibilities
  - Regression problem (predict continuous values)
  
- **Quantized:** 313 representative colors
  - Classification problem (predict one of 313 bins)
  - Each bin is a cluster center in ab space

**How it Works:**
1. **Analysis of Natural Images:**
   - Most natural images use limited color palette
   - Certain colors (sky blue, grass green) very common
   - Others (neon colors) rare

2. **K-means Clustering:**
   - Cluster ImageNet colors in ab space
   - Find 313 most representative colors
   - Saved in `pts_in_hull.npy`

3. **Prediction:**
   - Model outputs 313 probabilities
   - Weighted sum gives final ab values:
   \[
   (a, b) = \sum_{i=1}^{313} P(bin_i) \times (a_i, b_i)
   \]

**Advantages:**
- **Easier to Learn:** Classification easier than regression
- **Class Rebalancing:** Can weight rare colors higher during training
- **Faster Inference:** Softmax over 313 classes vs continuous optimization

**Q3: Why subtract 50 from the L channel before feeding to the network?**

**A3:** Subtracting 50 centers the L channel around zero for better neural network performance.

**L Channel Range:**
- Original: [0, 100] (black to white)
- After subtracting 50: [-50, 50] (centered at 0)

**Benefits:**

**1. Zero-Centered Data:**
- Activations start around zero
- Gradients flow more evenly
- Faster convergence

**2. Activation Function Efficiency:**
- Sigmoid/tanh centered at zero
- ReLU gets both positive and negative inputs

**3. Symmetry:**
- Treats darks and lights equally
- No bias toward bright or dark regions

**4. Numerical Stability:**
- Prevents saturated activations
- Reduces risk of exploding/vanishing gradients

**Mathematical Impact:**
```python
# Without centering
L = 90 (very bright) → neuron always active → gradient small

# With centering  
L = 90 - 50 = 40 → neuron sometimes active → better learning
```

---

### Technical Questions

**Q4: Walk through the model architecture. What layers are involved?**

**A4:** The colorization model uses an encoder-decoder architecture with skip connections.

**Encoder (Downsampling Path):**
```
Input: (224, 224, 1) - L channel
    ↓ Conv + ReLU + BN
(112, 112, 64)
    ↓ Conv + ReLU + BN
(56, 56, 128)
    ↓ Conv + ReLU + BN
(28, 28, 256)
    ↓ Conv + ReLU + BN
(14, 14, 512) - Bottleneck
```

**Decoder (Upsampling Path):**
```
(14, 14, 512)
    ↓ Deconv + ReLU + BN
(28, 28, 256) + skip connection
    ↓ Deconv + ReLU + BN
(56, 56, 128) + skip connection
    ↓ Conv (class8_ab layer)
(56, 56, 313) - Probabilities for 313 colors
    ↓ Softmax
(56, 56, 313)
    ↓ Conv (conv8_313_rh with pts_in_hull weights)
(56, 56, 2) - Final ab channels
```

**Key Layers:**

**1. class8_ab Layer:**
- Outputs 313 probability maps
- Each map represents one quantized color

**2. conv8_313_rh Layer:**
- Weighted sum of 313 color probabilities
- Weights are the 313 ab cluster centers
- Produces final ab prediction

**3. Skip Connections:**
- Connect encoder to decoder
- Preserve spatial detail lost in downsampling

**Q5: How does the model handle the "class8_ab" and "conv8_313_rh" layers?**

**A5:**

**class8_ab Layer:**
```python
# Layer configuration (from prototxt)
layer {
  name: "class8_ab"
  type: "Convolution"
  bottom: "conv8_313"
  top: "class8_ab"
  convolution_param {
    num_output: 313
    kernel_size: 1
    stride: 1
  }
}
```

**Purpose:**
- 1×1 convolution
- Outputs 313 probability maps (one per color bin)
- Each pixel gets distribution over 313 colors

**Code Setup:**
```python
# Load quantized ab values
numpy_file = np.load('./pts_in_hull.npy')  # Shape: (313, 2)
numpy_file = numpy_file.transpose().reshape(2, 313, 1, 1)

# Set as convolutional weights
Caffe_net.getLayer(Caffe_net.getLayerId('class8_ab')).blobs = [
    numpy_file.astype(np.float32)
]
```

**conv8_313_rh Layer:**
```python
# Set temperature/scaling parameter
Caffe_net.getLayer(Caffe_net.getLayerId('conv8_313_rh')).blobs = [
    np.full([1, 313], 2.606, np.float32)
]
```

**Purpose:**
- Weighted combination of 313 color bins
- Temperature parameter (2.606) controls color saturation
- Higher temperature → more vibrant colors

**Mathematical Operation:**
\[
\begin{bmatrix} a \\ b \end{bmatrix} = \sum_{i=1}^{313} \text{softmax}(z_i / T) \times \begin{bmatrix} a_i \\ b_i \end{bmatrix}
\]
where T = 2.606 is the temperature.

**Q6: What is the role of the temperature parameter (2.606)?**

**A6:** The temperature parameter controls color saturation in the output.

**Softmax with Temperature:**
\[
P_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
\]

**Effect of Temperature:**

**Low Temperature (T→0):**
- Sharper distribution
- More confident predictions
- Single color dominates
- **Result:** More saturated, vivid colors
- Risk: Overconfident, may produce unrealistic colors

**High Temperature (T→∞):**
- Smoother distribution
- Less confident predictions
- Multiple colors mixed
- **Result:** More muted, desaturated colors
- Risk: Washed-out appearance

**Chosen Value (T=2.606):**
- Empirically determined on ImageNet
- Balances saturation and realism
- Produces natural-looking colors

**Example:**
```python
# Without temperature (T=1)
logits = [2.0, 1.0, 0.5]
probs = softmax(logits)  # [0.66, 0.24, 0.10]

# With temperature (T=2.606)
probs = softmax(logits/2.606)  # [0.48, 0.30, 0.22] - smoother
```

---

### Implementation Questions

**Q7: Why resize the L channel to 224×224 but output is 56×56×2?**

**A7:**

**Input Resizing (224×224):**
- **Model Architecture:** Trained with 224×224 input
- **ImageNet Standard:** 224×224 is common size
- **Computational Efficiency:** Fixed size enables batch processing

**Output Size (56×56):**
- **Encoder-Decoder:** Multiple downsampling layers
- **Downsampling Factor:** 224 / 56 = 4× reduction
- **Design Choice:** Balance between detail and computation

**Why Not Full Resolution Output:**

**1. Computational Cost:**
- 224×224 output requires 16× more computation than 56×56
- Inference time: seconds → minutes

**2. Upsampling Strategy:**
- Bilinear interpolation is fast and effective
- Network learns coarse colors, upsampling adds detail

**3. Color Smoothness:**
- Colors vary slowly across image
- High-resolution color prediction unnecessary
- Details come from L channel (preserves edges)

**Upsampling Code:**
```python
# Output from network: (56, 56, 2)
ab_channel = Caffe_net.forward()[0,:,:,:].transpose((1,2,0))

# Upsample to original size
(original_height, original_width) = rgb_img.shape[:2]
ab_channel_us = cv.resize(ab_channel, (original_width, original_height))
```

**Result:**
- Sharp edges (from L channel at full resolution)
- Smooth colors (from ab channels at 56×56, upsampled)

**Q8: How would you modify this code to colorize a video?**

**A8:**

```python
import cv2 as cv
import numpy as np

# Load model (same as before)
Caffe_net = cv.dnn.readNetFromCaffe(
    "./models/colorization_deploy_v2.prototxt",
    "./models/colorization_release_v2.caffemodel"
)

# Setup model layers (same as before)
numpy_file = np.load('./pts_in_hull.npy')
numpy_file = numpy_file.transpose().reshape(2, 313, 1, 1)
Caffe_net.getLayer(Caffe_net.getLayerId('class8_ab')).blobs = [
    numpy_file.astype(np.float32)
]
Caffe_net.getLayer(Caffe_net.getLayerId('conv8_313_rh')).blobs = [
    np.full([1, 313], 2.606, np.float32)
]

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def colorize_frame(frame):
    """Colorize single frame."""
    # Convert to RGB and normalize
    rgb_img = (frame[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)
    
    # Convert to LAB
    lab_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2Lab)
    l_channel = lab_img[:,:,0]
    
    # Resize and normalize
    l_channel_resize = cv.resize(l_channel, (224, 224))
    l_channel_resize -= 50
    
    # Inference
    Caffe_net.setInput(cv.dnn.blobFromImage(l_channel_resize))
    ab_channel = Caffe_net.forward()[0,:,:,:].transpose((1,2,0))
    
    # Resize ab to original size
    (original_height, original_width) = rgb_img.shape[:2]
    ab_channel_us = cv.resize(ab_channel, (original_width, original_height))
    
    # Combine and convert
    lab_output = np.concatenate((l_channel[:,:,np.newaxis], ab_channel_us), axis=2)
    bgr_output = np.clip(cv.cvtColor(lab_output, cv.COLOR_Lab2BGR), 0, 1)
    
    return (bgr_output * 255).astype(np.uint8)


# Video colorization
input_video = cv.VideoCapture('input_video.mp4')
fps = int(input_video.get(cv.CAP_PROP_FPS))
width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))

# Output video writer
fourcc = cv.VideoWriter_fourcc(*'mp4v')
output_video = cv.VideoWriter('colorized_video.mp4', fourcc, fps, (width, height))

frame_count = 0
while True:
    ret, frame = input_video.read()
    if not ret:
        break
    
    # Colorize frame
    colorized_frame = colorize_frame(frame)
    
    # Write to output
    output_video.write(colorized_frame)
    
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

input_video.release()
output_video.release()
print("Video colorization complete!")
```

**Optimizations for Video:**

**1. Temporal Consistency:**
```python
# Smooth colors across frames
prev_ab = None
alpha = 0.7  # Smoothing factor

def colorize_frame_smooth(frame):
    global prev_ab
    
    ab_channel = colorize_frame(frame)  # Get current prediction
    
    if prev_ab is not None:
        # Exponential moving average
        ab_channel = alpha * ab_channel + (1 - alpha) * prev_ab
    
    prev_ab = ab_channel
    return ab_channel
```

**2. Batch Processing:**
```python
# Process multiple frames at once
batch_size = 4
frames_batch = []

while True:
    ret, frame = input_video.read()
    if not ret:
        break
    
    frames_batch.append(frame)
    
    if len(frames_batch) == batch_size:
        # Process batch (modify model to accept batch input)
        colorized_batch = colorize_batch(frames_batch)
        for cf in colorized_batch:
            output_video.write(cf)
        frames_batch = []
```

**3. GPU Acceleration:**
```python
# Use GPU if available
Caffe_net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
Caffe_net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
```

---

### Project-Specific Questions

**Q9: What are the limitations of this colorization approach?**

**A9:**

**1. Ambiguous Scenes:**
- **Problem:** Multiple valid colorizations
  ```
  - Red car vs blue car (both plausible)
  - Person's shirt color (infinite possibilities)
  ```
- **Model Behavior:** Tends toward average/common colors
- **Result:** Often produces desaturated, brownish tones

**2. Semantic Understanding:**
- **Problem:** Model doesn't understand objects
  ```
  - May color grass as blue
  - Sky might be green
  - Objects in unusual colors confused
  ```
- **Why:** Relies on local patterns, not global semantics

**3. Rare Colors:**
- **Problem:** 313 color bins biased toward common colors
  ```
  - Vivid reds/blues underrepresented
  - Neon colors not in bins
  - Unusual color combinations missing
  ```
- **Result:** Colors tend toward natural palette (browns, greens, blues)

**4. Context Dependency:**
- **Problem:** Same grayscale value can be different colors in different contexts
  ```
  - Gray circle: Could be metal, stone, cloud
  - Each should have different color
  ```
- **Model Limitation:** Limited context window

**5. Fine Details:**
- **Problem:** Output at 56×56, upsampled
  ```
  - Small objects may have incorrect colors
  - Color bleeding across boundaries
  - Text, patterns lose color detail
  ```

**Q10: How would you improve this colorization system?**

**A10:**

**1. User Guidance:**
```python
def colorize_with_hints(image, color_hints):
    """
    Allow user to specify colors for certain regions.
    
    color_hints: [(x, y, color), ...]
    """
    # Modify loss function to match specified colors
    # During inference, condition on user hints
    pass
```

**Example:**
- User clicks on sky → blue
- User clicks on dress → red
- Model fills in rest consistent with hints

**2. Semantic Segmentation:**
```python
# First, segment image into objects
segments = segment_image(image)  # {sky, grass, person, car}

# Colorize each segment with object-specific model
for segment in segments:
    if segment.type == 'sky':
        color = sky_colorization_model(segment)
    elif segment.type == 'grass':
        color = grass_colorization_model(segment)
    # ...
```

**Benefits:**
- Sky always blue
- Grass always green
- Context-appropriate colors

**3. GAN-based Colorization:**
```python
# Generator: Colorizes image
G(L) → (a, b)

# Discriminator: Judges if colorization realistic
D(L, a, b) → real/fake

# Adversarial training
Loss = L1_loss + λ * adversarial_loss
```

**Advantages:**
- More vibrant, realistic colors
- Better handling of ambiguity
- Captures complex color relationships

**4. Attention Mechanisms:**
```python
# Model learns to attend to relevant regions
attention_weights = attention_module(L_channel)
features = features * attention_weights
```

**Benefits:**
- Focus on informative regions
- Better long-range dependencies
- Improved semantic understanding

**5. Multi-Scale Processing:**
```python
# Process at multiple resolutions
colors_coarse = model_224(L_224)
colors_medium = model_448(L_448, colors_coarse)
colors_fine = model_896(L_896, colors_medium)
```

**Benefits:**
- Coarse: Global color scheme
- Medium: Object-level colors
- Fine: Detail preservation

**6. Reference-Based Colorization:**
```python
def colorize_with_reference(grayscale, reference_color_image):
    """
    Transfer colors from reference image.
    Match similar textures/patterns.
    """
    # Find corresponding regions
    # Transfer colors
    pass
```

**Example:**
- Colorize old family photo using recent photo of same person

---

## Additional Resources

**Papers:**
- Zhang et al. (2016): "Colorful Image Colorization" (Original paper)
- Iizuka et al. (2016): "Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors"
- Zhang et al. (2017): "Real-Time User-Guided Image Colorization with Learned Deep Priors"

**Datasets:**
- ImageNet: 1.3M images for training
- Places365: Scene-centric dataset
- COCO: Object-centric dataset

**Tools:**
- Colorization Demo: http://richzhang.github.io/colorization/
- DeOldify (GAN-based): https://github.com/jantic/DeOldify

