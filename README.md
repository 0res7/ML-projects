# ML-projects

## Repository Overview

This repository contains a collection of machine learning and AI projects, each in its own directory. Projects cover computer vision, NLP, tabular data, time series, and more. Most projects are implemented as Jupyter notebooks, with supporting scripts and documentation.

---

## Repository Architecture

```mermaid
graph TD;
A[ML-projects Repository] --> B[Project 1: Business Email Automation (LLM, RAG, Google Sheets)]
A --> C[Project 2: BRAIN TUMOR DETECTION]
A --> D[Project 3: Classification of Arrhythmia]
A --> E[Project 4: Colorize Black & white images]
A --> F[Project 5: Diabetes Prediction]
A --> G[Project 6: Distracted Driver Detection]
A --> H[Project 7: Drowsiness detection]
A --> I[Project 8: Gender and age detection]
A --> J[Project 9: Getting Admission in College Prediction]
A --> K[Project 10: Heart Disease Prediction]
A --> L[Project 11: Human Activity Detection]
A --> M[Project 12: IPL Score Prediction]
A --> N[Project 13: Iris Flower Classification]
A --> O[Project 14: Loan Repayment Prediction]
A --> P[Project 15: Mechanisms Of Action Prediction]
A --> Q[Project 16: Medical Chatbot]
A --> R[Project 17: Predicting Property Maintenance Fines]
A --> S[Project 18: Research topic Prediction]
A --> T[Project 19: Smile Selfie Capture Project]
A --> U[Project 20: TimeSeries Multi StoreSales prediction]
A --> V[Project 21: Wine Quality prediction]
A --> W[Project 22: HiringChallenges]
W --> W1[water_potability]
W --> W2[triglyceride_prediction]
A --> X[Project 23: AI Room Booking Chatbot]
A --> Y[...Other Projects]
```

---

## Project List

| Project Name | Description | Main Notebook |
|--------------|-------------|--------------|
| Business Email Automation (LLM, RAG, Google Sheets) | Business email classification and response automation using LLMs, RAG, and Google Sheets integration | [Copy of Solve Business Problems with AI.ipynb](AI-assessment/Copy%20of%20Solve%20Business%20Problems%20with%20AI.ipynb) |
| BRAIN TUMOR DETECTION [END 2 END] | Flask web app for brain tumor detection from MRI using PyTorch | (see project folder) |
| Classification of Arrhythmia [ECG DATA] | Arrhythmia classification using multiple ML models and PCA | [final with pca.ipynb](Classification%20of%20Arrhythmia%20%5BECG%20DATA%5D/final%20with%20pca.ipynb) |
| Colorize Black & white images [OPEN CV] | Image colorization using OpenCV and deep learning | [Colorize_Black_and_White_Image.ipynb](Colorize%20Black%20%26%20white%20images%20%5BOPEN%20CV%5D/Colorize_Black_and_White_Image.ipynb) |
| Diabetes Prediction [END 2 END] | Diabetes prediction and deployment as a web app | [Diabetes Classification.ipynb](Diabetes%20Prediction%20%5BEND%202%20END%5D/Diabetes%20Classification.ipynb) |
| Distracted Driver Detection | Driver activity classification from images | [Distrated Driver detection.ipynb](Distracted%20Driver%20Detection/Distrated%20Driver%20detection.ipynb) |
| Drowsiness detection [OPEN CV] | Real-time driver drowsiness detection using webcam and CNN | (see project folder) |
| Gender and age detection using deep learning | Age and gender prediction from images | (see project folder) |
| Getting Admission in College Prediction | Admission prediction using tabular data | [Admission prediction.ipynb](Getting%20Admission%20in%20College%20Prediction/Admission%20prediction.ipynb) |
| Heart Disease Prediction [END 2 END] | Heart disease prediction using ML | [Heart Disease Prediction.ipynb](Heart%20Disease%20Prediction%20%5BEND%202%20END%5D/Heart%20Disease%20Prediction.ipynb) |
| Human Activity Detection | Human action recognition using LSTM and Detectron2 | [Human_Activity_Recogination.ipynb](Human%20Activity%20Detection/Human_Activity_Recogination.ipynb) |
| IPL Score Prediction | First innings score prediction for IPL matches | [First Innings Score Prediction - IPL.ipynb](IPL%20Score%20Prediction/First%20Innings%20Score%20Prediction%20-%20IPL.ipynb) |
| Iris Flower Classification | Iris species classification using ML | [iris.ipynb](Iris%20Flower%20Classification/iris.ipynb) |
| Loan Repayment Prediction | Loan default prediction using ML | [Loan_Repayment_Prediction.ipynb](Loan%20Repayment%20Prediction/Loan_Repayment_Prediction.ipynb) |
| Mechanisms Of Action (MoA) Prediction | Multi-label drug response prediction | [MOA.ipynb](Mechanisms%20Of%20Action%20%28MoA%29%20Prediction/MOA.ipynb) |
| Medical Chatbot [END 2 END] [NLP] | End-to-end medical chatbot using NLP | [Meddy.ipynb](Medical%20Chatbot%20%5BEND%202%20END%5D%20%5BNLP%5D/Meddy.ipynb) |
| Predicting Property Maintenance Fines | Regression/classification for property fines | [Predicting Property Maintainance Fines.ipynb](Predicting%20Property%20Maintenance%20Fines/Predicting%20Property%20Maintainance%20Fines.ipynb) |
| Research topic Prediction | Research topic classification | [Research-topic-Prediction.ipynb](Research%20topic%20Prediction/Research-topic-Prediction.ipynb) |
| Smile Selfie Capture Project  [OPEN CV] | Smile detection and selfie capture | (see project folder) |
| TimeSeries Multi StoreSales prediction | Multi-store sales time series forecasting | [Time Series Regression - Multi-Store Sales.ipynb](TimeSeries%20Multi%20StoreSales%20prediction/Time%20Series%20Regression%20-%20Multi-Store%20Sales.ipynb) |
| Wine Quality prediction | Wine quality regression/classification | [Wine.ipynb](Wine%20Quality%20prediction/Wine.ipynb) |
| HiringChallenges/water_potability | Water potability prediction (hiring challenge) | (see project folder) |
| HiringChallenges/triglyceride_prediction | Triglyceride level regression (hiring challenge) | (see project folder) |
| AI Room Booking Chatbot [IBM WATSON] | Room booking chatbot using IBM Watson | (see project folder) |

---

## Project Architecture & Flow Diagrams

Below, each project will have a brief workflow/architecture diagram. (Diagrams will be added in subsequent updates.)

### Example (Business Email Automation)

```mermaid
flowchart TD
    A[Read Emails & Products from Google Sheets] --> B[Classify Emails with LLM]
    B --> C{Order Request?}
    C -- Yes --> D[Extract Order Details with LLM]
    D --> E[Check Stock & Update]
    E --> F[Generate Order Response with LLM]
    C -- No --> G[Product Inquiry]
    G --> H[Retrieve Relevant Products (RAG/FAISS)]
    H --> I[Generate Inquiry Response with LLM]
```

---

## Redundant Files & Cleanup Suggestions

- `.DS_Store`, `venv/`, and other environment or OS-generated files should be excluded via `.gitignore`.
- Some projects have duplicate scripts (e.g., `triglyceride_prediction/main.py` and `linear_regression.py` are nearly identical).
- Remove test scripts, outputs, or data files that are not needed for the main workflow.
- Consider consolidating similar notebooks/scripts within each project.
- Review large files (e.g., model weights, datasets) and use download scripts or links instead of storing them in the repo if possible.

---

## How to Run

Each project has its own README and requirements. See the project folder for details.

---

## Contributing

See `CONTRIBUTING.md` for guidelines.

---

## License

MIT License
=======
<div align="center">
  <img src="Resources/banner.png" alt="Banner" />
</div>



<div align="center">
  <h1><a href="https://github.com/0res7/ml-basics">Learn ML Basics</a></h1>
  <img alt="GIF" src="Resources/roll.gif" />
</div>


----------

## Project Overview

Welcome to the **Machine Learning Projects Repository**! This collection encompasses various projects demonstrating core concepts in **machine learning**, **deep learning**, **natural language processing (NLP)**, and **computer vision**. It includes both **deployed applications** (built using **Flask**) and **GUI-based apps** (using **Tkinter**). These projects illustrate the potential of machine learning across domains, including medical diagnosis, human activity recognition, image processing, and more.

## Project List

Here’s a detailed list of all projects included in this repository:

| Project Name                                | Description                                                                                                                                                    | Link |
|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| **AI Room Booking Chatbot**                 | An intelligent chatbot built with **IBM Watson Assistant** to facilitate room bookings.                                                                        | [AI Room Booking Chatbot](https://github.com/0res7/ML-projects/tree/predict-turnover/AI%20Room%20Booking%20Chatbot%20%5BIBM%20WATSON%5D) |
| **Predict Employee Turnover**               | Predict employee turnover using **scikit-learn** decision trees and random forest models.                                                                     | [Predict Employee Turnover](https://github.com/0res7/ML-projects/tree/predict-turnover/Predict%20Employee%20Turnover%20with%20scikitlearn) |
| **Wine Quality Prediction**                 | Predict wine quality using physicochemical features like acidity, sugar, and pH with machine learning models.                                                 | [Wine Quality Prediction](https://github.com/0res7/ML-projects/tree/predict-turnover/Wine%20Quality%20prediction) |


## Technologies Used

This repository includes a wide range of technologies and tools used in various machine learning and data science projects:

- **Programming Languages:** Python
- **Libraries/Frameworks:**
  - Machine Learning: scikit-learn, TensorFlow, PyTorch, Keras
  - NLP: IBM Watson, Natural Language Toolkit (NLTK), SpaCy
  - Web Development: Flask
  - Image Processing: OpenCV
  - GUI Development: Tkinter
  - Deep Learning: CNN, LSTM, DNN
- **Tools & Platforms:** 
  - IBM Watson, Google Colab, Jupyter Notebooks
  - Deployed apps using Flask
  - Git and GitHub for version control

<div align="center">
  <img alt="GIF" src="Resources/pac.gif" />
</div>

## 📊 Project Structure

Each project follows a consistent structure for easy navigation and understanding:
```plaintext
ProjectName/
│
├── data/                  # Data files and datasets
├── notebooks/             # Jupyter notebooks for experimentation and prototyping
├── models/                # Trained machine learning models (if applicable)
├── static/                # Static files (CSS, JS, images for Flask-based projects)
├── templates/             # HTML templates (for Flask-based projects)
├── src/                   # Core Python scripts for data preprocessing, model training, etc.
├── app.py                 # Main application file for Flask-based projects
├── README.md              # Project-specific readme file
└── requirements.txt       # List of dependencies for the project
```

Feel free to explore individual projects to understand the data flow and code structure.

---

<div align="center">
  <img alt="GIF" src="Resources/mario.gif" />
</div>

## 🌍 Deployment

Some of the projects can be easily deployed on cloud platforms like **Heroku**, **AWS**, or **Azure**. The following steps outline a generic approach for deploying a Flask-based web app on Heroku:

1. **Install Heroku CLI**:  
   Follow the instructions [here](https://devcenter.heroku.com/articles/heroku-cli).

2. **Login to Heroku**:  
   ```bash
   heroku login
   ```

3. **Create a new Heroku app**:  
   ```bash
   heroku create your-app-name
   ```

4. **Push to Heroku**:  
   Ensure your `Procfile` is correctly set up for Flask:
   ```plaintext
   web: gunicorn app:app
   ```
   Then push the project to Heroku:
   ```bash
   git push heroku main
   ```

5. **View your deployed app**:  
   ```bash
   heroku open
   ```

You can follow similar steps for AWS (using **Elastic Beanstalk**) or Azure (using **App Services**).


## 📚 Resources and References

<div align="center">
  <img alt="GIF" src="Resources/python.gif" />
</div>


- **Official Python Documentation**: [Python.org](https://docs.python.org/3/)
- **Flask Documentation**: [Flask.palletsprojects.com](https://flask.palletsprojects.com/en/2.0.x/)
- **Scikit-learn User Guide**: [Scikit-learn.org](https://scikit-learn.org/stable/user_guide.html)
- **Keras Documentation**: [Keras.io](https://keras.io/)
- **TensorFlow Documentation**: [Tensorflow.org](https://www.tensorflow.org/)
- **PyTorch Documentation**: [Pytorch.org](https://pytorch.org/docs/)

For a deeper understanding of AI, machine learning, and data science, I recommend the following courses:
- **Coursera : What is Data Science?**
- **Kaggle Learn : Data Science**
- **Jovian AI : Beginner-Friendly**

## ⭐ Acknowledgments

- The wonderful **Kaggle** community, which provided open datasets and insightful discussions.
- **Jovian**, **Coursera**, and **edX** instructors who have helped me build a solid foundation in AI.
- **ChatGPT** and its GPT buddies, for answering all my silly questions and being my AI BFFs 🤖😂💡, keeping me company when coffee couldn't ☕😉!


<div align="center">
  <img alt="GIF" src="Resources/busy-work.gif" />
</div>

