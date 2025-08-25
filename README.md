# Employee Absenteeism Prediction & Visualization

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Visualizations & Insights](#-visualizations--insights)
3. [Machine Learning Model](#-machine-learning-model)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [How to Use](#-how-to-use)

---

### 📌 Overview
This project provides a comprehensive solution for understanding and predicting employee absenteeism. It combines exploratory data analysis using a Tableau dashboard with a predictive machine learning model built in Python. The goal is to deliver actionable insights that can help HR departments manage their workforce more effectively.

### 📊 Visualizations & Insights

#### Tableau Dashboard
An interactive Tableau dashboard was created for a high-level overview of absenteeism trends. This allows for exploration of absence reasons, seasonality, and demographic impacts.

![Tableau Absenteeism Dashboard](visualizations/tableau_absenteeism_dashboard.png)

#### Key Model Insights
After training the model, I analyzed the relationship between key features and the predicted probability of excessive absenteeism.

**1. Probability of Absenteeism by Reason**
The various reasons for absence were grouped into four categories. The model shows that "Reason 1" (various diseases) and "Reason 3" (poisoning or specific conditions) have a significantly higher probability of leading to long-term absence compared to others.

![Reasons vs. Probability Plot](visualizations/reasons_vs_probability.png)
<img width="1366" height="768" alt="reasons_vs_probability" src="https://github.com/user-attachments/assets/7ae3fba5-5cf9-4127-869f-6200e75f5e8c" />


**2. Probability of Absenteeism by Age**
The analysis reveals a non-linear relationship between age and the probability of excessive absence. The probability tends to increase through an employee's late 20s and 30s, peaking around age 40 before declining.

![Age vs. Probability Plot](visualizations/age_vs_probability.png)
<img width="1366" height="768" alt="age_vs_probability" src="https://github.com/user-attachments/assets/37c3b0bb-5572-42b0-96a0-2246ed8eab4a" />

### 🤖 Machine Learning Model
A Logistic Regression model was trained to predict whether an employee's absence would be longer than the median duration.

-   **Model:** `Logistic Regression`
-   **Performance:** **77.14% Accuracy** on the hold-out test set.

The model is encapsulated in `absenteeism_module.py` for easy integration and prediction on new data.

### 💻 Tech Stack
-   **Machine Learning:** Python, Pandas, NumPy, Scikit-learn
-   **Data Visualization:** Tableau, Matplotlib/Seaborn
-   **Development:** Jupyter Notebook, Git & GitHub
-   
## 🚀 How to Use

1.  **Clone the repository:**
    ```bash
    git clone (https://github.com/sayankumarnandi2003/Employee-Absenteeism-Prediction).git)
    cd Employee-Absenteeism-Prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn
    ```

3.  **Train the Model:**
    Run the `train_model.py` script. This will use `Absenteeism_preprocessed (1).csv` to create two new files in the main folder: `model` and `scaler`.
    ```bash
    python train_model.py
    ```

4.  **Make New Predictions:**
    Open and run the Jupyter Notebook (`predictions2108.ipynb`). It will use the `model` and `scaler` files to predict absenteeism for the data in `Absenteeism_new_data.csv` and will save the results to `Absenteeism_predictions.csv`.

## 📈 Results
The final Logistic Regression model achieves an accuracy of **77.14%** on the test set, demonstrating a strong ability to predict excessive absenteeism.

### 📂 Project Structure
├── data/
├── saved_models/
├── notebooks/
├── visualizations/
│   ├── tableau_absenteeism_dashboard.png
│   ├── age_vs_probability.png
│   └── reasons_vs_probability.png
├── absenteeism_module.py
├── train_model.py
└── README.md


