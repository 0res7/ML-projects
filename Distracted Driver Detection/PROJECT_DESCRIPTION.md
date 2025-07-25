
# Project Description: Distracted Driver Detection

## Objective
The main objective of this project is to develop a deep learning model that can classify images of drivers into one of ten categories, representing different levels of distraction. The goal is to accurately predict what the driver is doing in each picture, such as safe driving, texting, talking on the phone, or reaching behind.

## Dataset Used
The project uses a dataset of driver images, which are categorized into the following ten classes:

- **c0**: safe driving
- **c1**: texting - right
- **c2**: talking on the phone - right
- **c3**: texting - left
- **c4**: talking on the phone - left
- **c5**: operating the radio
- **c6**: drinking
- **c7**: reaching behind
- **c8**: hair and makeup
- **c9**: talking to passenger

The dataset is split into training and testing sets, and the images are preprocessed before being fed into the model.

## ML Pipeline
The machine learning pipeline in this project can be summarized as follows:

1.  **Data Preprocessing**:
    - The images are loaded from the disk and resized to a fixed size of 224x224 pixels.
    - The pixel values are normalized to be between 0 and 1.
    - The labels are one-hot encoded.

2.  **Model Architecture**: The project uses a pre-trained ResNet-50 model as the base for the classifier. The final fully connected layer of the ResNet-50 model is replaced with a custom classifier to adapt it to the specific task of distracted driver detection. The custom classifier consists of several dense layers with ReLU activation functions and a final softmax layer for classification.

3.  **Model Training**: The model is trained for 10 epochs with a batch size of 32. The Adam optimizer is used to minimize the categorical cross-entropy loss function. The model's performance is evaluated on a validation set during training.

4.  **Prediction**: The trained model is used to make predictions on the test set. The class with the highest probability is chosen as the final prediction.

## Technical Stack
- **Programming Language**: Python
- **Deep Learning Framework**: Keras with TensorFlow backend
- **Libraries**:
    - **NumPy**: For numerical operations.
    - **Pandas**: For data manipulation and analysis.
    - **Matplotlib**: For data visualization.
    - **OpenCV**: For image processing.
    - **Scikit-learn**: For data preprocessing and evaluation.

## Key Insights
- The use of a pre-trained ResNet-50 model with transfer learning is an effective approach for this image classification task.
- The model achieves a high accuracy on the validation set, indicating that it is able to learn the distinguishing features of the different classes.
- The confusion matrix shows that the model is able to correctly classify most of the instances, with some confusion between similar classes, such as texting and talking on the phone.
