
# Project Description: Brain Tumor Detection

## Objective
The primary objective of this project is to develop a deep learning model that can accurately classify different types of brain tumors from MRI images. The model is deployed as a web application, allowing users to upload an MRI image and receive a prediction about the type of brain tumor present.

## Dataset Used
The project uses a dataset of brain MRI images, which are categorized into four classes:
- **Glioma**: A type of tumor that occurs in the brain and spinal cord.
- **Meningioma**: A tumor that arises from the meninges — the membranes that surround your brain and spinal cord.
- **Pituitary**: Tumors that develop in the pituitary gland.
- **No Tumor**: Healthy brain scans.

The model is trained on these images to learn the distinguishing features of each class.

## ML Pipeline
The machine learning pipeline for this project is as follows:

1.  **Data Preprocessing**: The MRI images are preprocessed before being fed into the model. This includes:
    - **Resizing**: The images are resized to a fixed size of 512x512 pixels to ensure consistency.
    - **Conversion to Tensor**: The images are converted into PyTorch tensors, which are the primary data structure used in PyTorch.

2.  **Model Architecture**: The project utilizes a pre-trained ResNet-50 model, a powerful convolutional neural network (CNN) architecture, as the base for the classifier. The final fully connected layer of the ResNet-50 model is replaced with a custom classifier to adapt it to the specific task of brain tumor classification. The custom classifier consists of several linear layers with SELU activation functions and dropout for regularization.

3.  **Model Training**: The pre-trained ResNet-50 model is fine-tuned on the brain tumor dataset. The model is trained to minimize the loss function, which is likely a cross-entropy loss, and the weights are updated using an optimizer like Adam or SGD.

4.  **Prediction**: For a given MRI image, the model outputs a prediction, which is the class with the highest probability. The output is one of the four classes: Glioma, Meningioma, Pituitary, or No Tumor.

## Web Application
The project includes a web application built with Flask, a popular Python web framework. The application allows users to:
- Upload an MRI image of a brain scan.
- The application then processes the image, feeds it to the trained deep learning model, and displays the predicted tumor type to the user.

The user interface is created using HTML templates, and the application is designed to be user-friendly and intuitive.

## Technical Stack
- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Web Framework**: Flask
- **Libraries**:
    - **Pillow (PIL)**: For image manipulation.
    - **Torchvision**: For image transformations and pre-trained models.
    - **os**: For interacting with the operating system.
