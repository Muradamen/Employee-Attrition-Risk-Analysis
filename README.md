# 🚀 Strategic Employee Attrition Risk Analytics

## 📌 Executive Summary

Employee attrition is a critical business risk that directly impacts productivity, talent retention, and operational costs. Most organizations react to attrition after employees leave — resulting in increased hiring costs and knowledge loss.

This project leverages **Machine Learning and People Analytics** to predict employee attrition and identify key risk drivers, enabling organizations to take **proactive, data-driven retention actions**.

---
Problem Statement:

Employee attrition is a major business challenge impacting productivity,
organizational knowledge, and operational costs.

Traditional HR approaches are reactive — addressing attrition only after 
employees leave. This leads to increased hiring costs and workforce instability.
## 🎯 Business Objectives

* Predict employees at high risk of attrition
* Identify key drivers behind employee turnover
* Enable proactive HR decision-making
* Reduce costs associated with recruitment and training

---

## 📊 Dataset

This project uses the **IBM HR Analytics Employee Attrition Dataset**, which includes rich employee-level data across:

* **Demographics:** Age, Gender
* **Financials:** MonthlyIncome
* **Work Metrics:** YearsAtCompany, JobRole, Department, OverTime
* **Behavioral Factors:** JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance
* **Other Factors:** DistanceFromHome, Education, MaritalStatus

🎯 **Target Variable:**

* Attrition (0 = Stayed, 1 = Left)

---

## ⚙️ Project Workflow

### 🔹 1. Data Preprocessing

* Cleaned dataset and removed irrelevant columns
* Encoded categorical variables using Label Encoding
* Transformed target variable (Yes/No → 1/0)
* Performed 80/20 train-test split (`random_state=42`)

---

### 🔹 2. Exploratory Data Analysis (EDA)

#### 📌 Attrition Distribution

![Attrition Distribution](images/attrition_distribution.png)

#### 📌 Monthly Income vs Attrition

![Income vs Attrition](images/income_attrition.png)

#### 📌 Overtime Impact

![Overtime vs Attrition](images/overtime_attrition.png)

#### 📌 Distance from Home vs Attrition

![Distance vs Attrition](images/distance_attrition.png)

---

## 🤖 Models Implemented

| Model               | Purpose                     |
| ------------------- | --------------------------- |
| Logistic Regression | Baseline (interpretability) |
| Decision Tree       | Explainable decision paths  |
| Random Forest       | Production-ready model      |

---

## 📈 Model Evaluation

Models were evaluated using:

* Accuracy
* F1 Score
* Confusion Matrix

#### 📊 Confusion Matrix Comparison

![Confusion Matrix](images/confusion_matrix.png)

---

## 🔍 Feature Importance (Attrition Drivers)

![Feature Importance](images/feature_importance.png)

### 🔑 Top Predictors:

* MonthlyIncome
* Age
* OverTime
* DistanceFromHome
* JobSatisfaction

---

## 🧠 Key Insights (Advanced Analysis)

* 💰 **Low Income → High Attrition Risk**
* ⏱ **OverTime → Burnout Indicator**
* 🚗 **Long Commute → Hidden Risk Factor**
* 😊 **Low Job Satisfaction → Strong Predictor of Exit**
* 👶 **Early Career Employees → Higher Turnover**

---

## 💼 Business Recommendations

### 1. Compensation Strategy

Adjust salary structures for high-risk employees.

### 2. Burnout Reduction

Monitor overtime and optimize workload distribution.

### 3. Retention Programs

Focus on early-tenure employees with targeted engagement strategies.

### 4. Workplace Experience

Improve job satisfaction and work-life balance initiatives.

---

## 🚀 Business Impact

* Reduce employee turnover
* Lower hiring and onboarding costs
* Improve workforce stability
* Enable data-driven HR decision-making

---

## 🧠 Logistic Regression Insight

Logistic Regression provided strong interpretability, allowing clear understanding of feature influence on attrition probability. However, due to its linear assumptions, it underperformed compared to Random Forest.

👉 Best used for **explainability**, not production deployment.

---

## 🧰 Tech Stack

* **Python**
* **pandas, numpy**
* **scikit-learn**
* **matplotlib, seaborn**

---

## 🔮 Future Enhancements

* SHAP explainability for model transparency
* Streamlit dashboard for HR visualization
* Integration with HR systems for real-time predictions

---

## 📁 Project Structure

```
Employee-Attrition-Risk-Analysis/
│
├── notebooks/
│   └── attrition_analysis.ipynb
│
├── images/
│   ├── attrition_distribution.png
│   ├── income_attrition.png
│   ├── overtime_attrition.png
│   ├── distance_attrition.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
└── README.md
```

---

## 👤 Author

**Murad Amin**
**Murad Amin** 🔗 [LinkedIn](https://www.linkedin.com/in/muradamin) | 🔗 [GitHub](https://github.com/Muradamen)

Data Scientist | AI & People Analytics
Passionate about building intelligent systems that solve real-world business problems.


## ⭐ If you found this project useful, give it a star!
