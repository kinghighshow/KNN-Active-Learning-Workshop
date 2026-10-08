# 🧑‍💻 K-Nearest Neighbors (KNN) Workshop

## 📘 Workshop Summary
This repository contains the materials and student solutions for the **K-Nearest Neighbors (KNN) Workshop**.  
The workshop introduces students to data analysis, machine learning design patterns, and active learning practices through the implementation of the KNN algorithm.

Students will explore multiple types of datasets (CSV files, APIs, relational databases), implement KNN classifiers, and design their code using the **Machine Learning Pipeline Pattern**. The workshop emphasizes collaboration through **peer programming** and reflection on algorithm performance.

---

## 🎯 Learning Objectives
By completing this workshop, you will be able to:
1. Understand the intuition behind the K-Nearest Neighbors algorithm.  
2. Acquire, clean, and preprocess data from different sources.  
3. Apply KNN for classification and evaluate its performance.  
4. Architect code using the **Machine Learning Pipeline Pattern**.  
5. Collaborate with peers through active learning and peer programming.  
6. Reflect on the strengths, limitations, and real-world applications of KNN.  

---

## 🤖 About the KNN Algorithm
The **K-Nearest Neighbors algorithm** is a simple, non-parametric machine learning method used for classification and regression.  
- A new data point is classified based on the **majority label of its *k* closest neighbors** in the training set.  
- The choice of *k* and the method of measuring distance (e.g., Euclidean, Manhattan) heavily influence performance.  
- KNN is **instance-based** (lazy learning): it stores the training data and makes predictions only at query time.  

---

## 🏗️ Implementation with ML Architecture and Design Patterns
This workshop emphasizes implementing KNN using the **Machine Learning Pipeline Pattern**:
- **Data Acquisition** (from CSV, API, or relational database)  
- **Data Cleaning & Preprocessing** (scaling, encoding, handling missing values)  
- **Feature Engineering**  
- **Model Training** (KNN classifier)  
- **Evaluation** (accuracy, confusion matrix, other metrics)  
- **Reflection & Peer Comparison**  

By structuring code into **modular components**, students will learn how to build reproducible and extensible ML workflows, aligning with professional MLOps practices.

---

## 📂 High-Level View of the Workshop
The workshop is split across two sessions (two hours total), with homework in between:

- **Session 1**  
  - Introduction to KNN (theory and Iris dataset demo)  
  - Hands-on: load and preprocess a dataset from a CSV, API, or relational DB  

- **Homework**  
  - Finalize preprocessing and prepare dataset for KNN  

- **Session 2**  
  - Implement KNN using the Pipeline Pattern  
  - Train, evaluate, and visualize results  
  - Reflect on algorithm performance  
  - Compare results with peers and discuss strengths/weaknesses of KNN  

---

## 🛠️ Local Setup

This project uses Python 3.14 and the notebook dependencies listed in `requirements.txt`.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

---

## 📝 Deliverables
- `KNN_Workshop_Solution.ipynb`: Your notebook implementation with code and explanations.  
- `README.md`: This file, including your name, student ID, and a brief summary of your work.  
- A PDF file with your name, student ID, and the URL to your remote repository.  

---

## ✍️ Instructor Information
- **Name:** *David Espinosa*  
- **Release:** *October 2025*  
