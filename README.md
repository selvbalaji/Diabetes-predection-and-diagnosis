Diabetic Prediction and Diagnosis System
Overview
This project implements a machine learning-based system for predicting diabetes risk and diagnosing potential cases using multiple algorithms. The system is built with Flask for the backend and HTML/CSS/JS for the frontend, providing a user-friendly interface for interacting with the models.

Features
Machine Learning Models: Utilizes three machine learning algorithms:
Logistic Regression: For binary classification of diabetes risk.
Linear Regression: To predict diabetes risk, with a threshold to convert predictions into binary values.
XGBoost: A powerful gradient boosting algorithm for classification.
Data Preprocessing: Includes steps such as handling missing values, feature scaling, and using SMOTE for addressing class imbalance.
Web Interface:
Prediction Form: Allows users to input personal and health-related information.
Results Page: Displays detailed results, predictions, and recommendations based on user input.
Project Structure
app.py: The main Flask application file that handles routing and prediction logic.
model_training.py: Contains the code for training and evaluating machine learning models.
preprocessing.py: Handles data preprocessing steps.
templates/: Directory for HTML templates:
index.html: The form for inputting data.
result.html: Displays prediction results and recommendations.
static/: Directory for CSS and JavaScript files:
style.css: CSS file for styling the frontend pages.
result.css: CSS file for styling the results page.
models/: Directory where trained models and scaler are saved.
requirements.txt: Lists the Python packages required for the project.
Installation
Clone the Repository:

bash
Copy code
git clone https://github.com/yourusername/diabetic-prediction.git
cd diabetic-prediction
Install Dependencies:
Ensure you have Python 3.6+ installed, then install the required packages:

bash
Copy code
pip install -r requirements.txt
Run the Application:
Start the Flask application:

bash
Copy code
python app.py
Open a web browser and navigate to http://127.0.0.1:5000 to access the prediction form.

Usage
Input Data: Fill out the form with the required information and submit.
View Results: After submission, the results page will display detailed results and recommendations based on the prediction.
Contributing
Feel free to submit issues or pull requests. Contributions are welcome to improve the functionality and accuracy of the system.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Flask: For creating the web application.
Scikit-learn: For machine learning algorithms and preprocessing tools.
XGBoost: For the gradient boosting algorithm.
