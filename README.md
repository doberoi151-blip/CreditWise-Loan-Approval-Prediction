# CreditWise-Loan-Approval-Prediction
A machine learning project that explores applicant and financial attributes to predict whether a loan application is approved.
Project type: Classification | Status: Notebook-based project

Project Overview
This project uses a loan approval dataset to explore patterns in applicant information and train classification models for the Loan_Approved target.

The notebook covers data inspection, missing-value treatment, exploratory data analysis (EDA), categorical encoding, feature scaling, model training, evaluation, and a small feature-engineering experiment.
Dataset
The notebook loads a file named loan_approval_data.csv.

The dataset is expected to include fields used in the notebook, such as:
- Applicant_ID
- Gender
- Education_Level
- Employment_Status
- Marital_Status
- Loan_Purpose
- Property_Area
- Employer_Category
- Applicant_Income
- Coapplicant_Income
- Credit_Score
- DTI_Ratio
- Savings
- Loan_Approved
The CSV file was not included with the uploaded notebook. Add the dataset to the project folder before running the notebook, or update the CSV path in the first data-loading section.
Workflow
1. Data inspection – load the CSV and inspect the data, summary statistics, and missing values.
2. Missing-value handling – use mean imputation for numerical columns and most-frequent imputation for categorical columns.
3. Exploratory data analysis – examine the target distribution, education levels, income distributions, and relationships between selected financial variables and loan approval.
4. Feature preparation – remove Applicant_ID, label-encode Education_Level and Loan_Approved, and one-hot encode selected categorical columns.
5. Correlation analysis – calculate and visualize correlations among numerical features.
6. Train/test split and scaling – use an 80/20 split with random_state=42, then standardize features using StandardScaler.
7. Model training and evaluation – train Logistic Regression, K-Nearest Neighbors (KNN), and Gaussian Naive Bayes.
8. Feature engineering experiment – create squared DTI_Ratio and Credit_Score features, remove the corresponding original columns from the feature matrix, and evaluate the models again.
Models and Notebook Results
The notebook reports the following metrics from its saved outputs.
Initial feature set
Model	Accuracy	Precision	Recall	F1-score
Logistic Regression	86.50%	78.33%	77.05%	77.69%
KNN (k=5)	76.00%	62.75%	52.46%	57.14%
Gaussian Naive Bayes	86.50%	80.36%	73.77%	76.92%


After feature engineering
Model	Accuracy	Precision	Recall	F1-score
Logistic Regression	88.00%	78.46%	83.61%	80.95%
KNN (k=5)	78.50%	67.31%	57.38%	61.95%
Gaussian Naive Bayes	86.00%	81.13%	70.49%	75.44%


These are the values stored in the notebook outputs; results may differ when the notebook is rerun with a different dataset or environment.
Technologies Used
- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
How to Run
1. Clone or download this repository.
2. Place loan_approval_data.csv in the same directory as the notebook, or update the path in the data-loading cell.
3. Install the dependencies:
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
4. Launch Jupyter:
   jupyter notebook
5. Open credit_wise.ipynb and run the cells in order.
Important Notes and Future Improvements
- Preprocessing leakage: In the current notebook, imputation and categorical encoding are performed before the train/test split. For a more reliable evaluation, fit preprocessing steps only on the training data, ideally using a Scikit-learn Pipeline and ColumnTransformer.
- Feature engineering evaluation: The notebook performs a second train/test split for the engineered features. A fixed split and consistent preprocessing pipeline would make model comparisons more reliable.
- Reproducibility: The split uses random_state=42; consider also using stratify=y to preserve class proportions.
- Model selection: The notebook includes a note selecting Naive Bayes based on precision in the initial experiment. After feature engineering, Logistic Regression has the highest reported F1-score and accuracy, while Naive Bayes has the highest precision. Choose the model based on the project's actual objective and error trade-offs.
- Add a requirements file, dataset source/licence, and a saved evaluation report when available.
Repository Contents
.
├── credit_wise.ipynb
├── loan_approval_data.csv   # add this dataset locally
└── README.md
Author
Dev Srivastava      
