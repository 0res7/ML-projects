<div align="center">
  <img src="banner.png" alt="Banner" />
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
- **Coursera - Machine Learning by Andrew Ng**
- **Udacity - AI for Everyone**
- **Kaggle Learn - Data Science**

## ⭐ Acknowledgments

- The wonderful **Kaggle** community, which provided open datasets and insightful discussions.
- **Udemy**, **Coursera**, and **edX** instructors who have helped me build a solid foundation in AI.


<div align="center">
  <img alt="GIF" src="Resources/busy-work.gif" />
</div>
