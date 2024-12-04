---

# **CodeHabor: An AI-Powered Code Analysis Tool**

Welcome to the **CodeHabor** repository! This project is a comprehensive, AI-driven solution designed to assist developers in managing code complexity, improving code quality, and enhancing accessibility. With four integrated components, CodeHabor addresses modern software development challenges using cutting-edge technologies and machine learning models.

---

## **Project Overview**

CodeHabor is a collaborative research project focused on managing code complexity and promoting better coding practices. The tool integrates AI and ML techniques to deliver actionable recommendations, refactoring suggestions, and accessibility improvements. It supports languages like **Python**, **Java**, and **JavaScript** and provides a gamified experience for users to enhance their coding practices.

---

## **Components**

### **1. Automated Code Review and Compliance Checking**
This component focuses on:
- Detecting code issues and standard violations using AI models.
- Providing recommendations for improvement based on coding standards.
- Training an AI-driven recommendation engine to generate context-aware suggestions.

### **2. AI Integration for Complexity Reduction**
This component utilizes AI techniques to:
- Suggest refactoring opportunities for reducing code complexity.
- Identify redundant or inefficient code and recommend optimized solutions.
- Employ machine learning models trained on labeled code samples.

### **3. Logic for Calculating and Displaying Code Complexity**
Key functionality includes:
- Measuring **Cyclomatic Complexity** and other metrics.
- Displaying metrics such as the number of functions, variables, loops, and average function length.
- Visualizing complexity levels to help developers better understand their code structure.

### **4. Enhancing Accessibility with Gamification**
Features include:
- Making code accessible through gamified tests and challenges.
- Encouraging developers to improve code accessibility using engaging, interactive tools.
- Focusing on inclusivity by detecting potential accessibility issues in code.

---

## **Technologies Used**

- **Frontend:** React, HTML
- **Backend:** Django
- **Machine Learning:** scikit-learn, PyTorch, Transformers (CodeT5), Random Forest
- **Database:** MongoDB
- **Tools:** Google Colab, PyCharm, VS Code

---

## **Dataset and Model Training**

### **Datasets**
- Custom and publicly available datasets from sources like **HuggingFace** and **GitHub**.
- Additional datasets for **Cyclomatic Complexity Metrics** and labeled code samples for secure and insecure code.

### **Model Training**
- CodeT5: Used for code understanding ang generating.
- Random Forest: Employed for calculating metrics and detecting defects based on structured code attributes (e.g., cyclomatic complexity, number of functions).

---

## **Installation and Setup**

1. Clone the repository:
   ```bash
   git clone https://github.com/yasidew/Codeharbor-2.0
   ```
2. Navigate to the project directory:
   ```bash
   cd CodeHabor
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the backend server:
   ```bash
   python manage.py runserver
   ```

---
