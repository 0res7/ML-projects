
# Project Description: Iris Flower Classification

## Objective
The main goal of this project is to classify iris flowers into one of three species (Iris-setosa, Iris-versicolor, and Iris-virginica) based on the measurements of their sepals and petals. This is a classic multi-class classification problem in machine learning.

## Dataset Used
The project utilizes the well-known Iris dataset. The dataset consists of 150 samples, with 50 samples for each of the three species. Four features were measured from each sample:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

The notebooks `[iris.ipynb](./iris.ipynb)` and `[SVM Iris.ipynb](./SVM Iris.ipynb)` load the dataset, either from a local CSV file or directly from the `sklearn.datasets` library.

## ML Pipeline
The machine learning pipeline in this project can be summarized as follows:

1.  **Data Loading and Exploration**: The data is loaded into a pandas DataFrame. Exploratory Data Analysis (EDA) is performed using various visualization techniques like histograms and scatter plots to understand the data distribution and feature correlations. The `[iris.ipynb](./iris.ipynb)` notebook provides a comprehensive EDA.

2.  **Data Preprocessing**:
    - **Label Encoding**: The categorical target variable, "Species," is converted into numerical labels (0, 1, 2) using `sklearn.preprocessing.LabelEncoder`.
    - **Feature Selection**: In `[iris.ipynb](./iris.ipynb)`, different combinations of features are used for training the models. In `[SVM Iris.ipynb](./SVM Iris.ipynb)`, only the sepal length and sepal width are used to visualize the decision boundaries of the SVM classifiers.
    - **Train-Test Split**: The dataset is split into training and testing sets to evaluate the performance of the models on unseen data.
    - **Feature Scaling**: `sklearn.preprocessing.StandardScaler` is used to standardize the features, which is important for algorithms like SVM and KNN.

3.  **Model Training and Evaluation**:
    - The `[iris.ipynb](./iris.ipynb)` notebook trains and evaluates several classification models:
        - Decision Tree Classifier
        - K-Nearest Neighbors (KNN)
        - Support Vector Machine (SVM)
        - Logistic Regression
        - Naive Bayes
    - The `[SVM Iris.ipynb](./SVM Iris.ipynb)` notebook specifically explores the SVM algorithm with different kernels (linear, polynomial, and RBF) and hyperparameters (degree, C, and gamma) to visualize their impact on the decision boundaries.
    - The models are evaluated based on accuracy, RMSE, and R2-score. A classification report is also generated for the Decision Tree Classifier.

## Technical Stack
- **Programming Language**: Python
- **Libraries**:
    - **Pandas**: For data manipulation and analysis.
    - **NumPy**: For numerical operations.
    - **Matplotlib & Seaborn**: For data visualization.
    - **Scikit-learn**: For implementing machine learning algorithms, preprocessing, and evaluation.

## Key Insights
- The Iris dataset is a well-behaved dataset with clearly separable classes, especially Iris-setosa.
- The EDA reveals strong correlations between petal length and petal width, and these two features are highly effective in distinguishing between the species.
- The SVM with an RBF kernel, Logistic Regression, and Naive Bayes models all achieve very high accuracy (around 98%) on the test set, indicating their suitability for this classification task.
- The visualizations in `[SVM Iris.ipynb](./SVM Iris.ipynb)` provide an excellent understanding of how different SVM kernels and their hyperparameters affect the model's decision boundary and its ability to separate the classes.
