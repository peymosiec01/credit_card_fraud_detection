**Credit Card Fraud Detection**

**Description:**
This repository contains code for a machine learning model designed to detect fraudulent credit card transactions. The model is trained on a simulated credit card transaction dataset spanning from January 1st, 2019 to December 31st, 2020. The dataset includes information about transactions, such as transaction date and time, credit card number, merchant details, transaction amount, and whether the transaction is fraudulent or legitimate.

**Files:**
1. `credit_card_fraud_detection.ipynb`: Jupyter Notebook containing the code for the machine learning model.
2. `README.md`: Readme file providing an overview of the project and instructions for running the code.
3. `fraudTrain.csv`: Simulated credit card transaction dataset used for training the model.
4. `fraudTest.csv`: Simulated credit card transaction dataset used for testing the model.

**Results Summary:**

| Model              | Accuracy | Precision | Recall | F1 Score | AUC      |
|--------------------|----------|-----------|--------|----------|----------|
| Naive Random Forest| 0.96     | 0.11      | 0.76   | 0.19     | 0.868    |
| Naive Decision Tree| 0.97     | 0.15      | 0.96   | 0.26     | 0.995    |
| Naive Logistic Reg.| 0.95     | 0.1       | 0.96   | 0.18     | 0.957    |
| Fine-tuned RF      | 0.97     | 0.15      | 0.75   | 0.25     | 0.845    |
| Fine-tuned DT      | 0.97     | 0.14      | 0.77   | 0.24     | 0.950    |
| Fine-tuned LR      | 0.97     | 0.14      | 0.77   | 0.24     | 0.876    |
| AdaBoost on RF     | 0.96     | 0.14      | 0.96   | 0.24     | 0.994    |
| Test Set Evaluation| 0.97     | 0.11      | 0.9    | 0.19     | 0.989    |

**Conclusion:**
While the model demonstrates high accuracy, it shows poor performance in terms of precision and F1 score, especially considering the class imbalance. Further optimization and fine-tuning of hyperparameters may improve performance.

**Instructions:**
To run the code:
1. Clone the repository to your local machine.
2. Open the `credit_card_fraud_detection.ipynb` notebook using Jupyter Notebook or any compatible environment.
3. Run the code cells sequentially to train and evaluate the machine learning model.
   
## Acknowledgements

- The SMS Spam Collection dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data).
- Libraries used: NumPy, pandas, scikit-learn, spaCy, seaborn, Matplotlib, imbalanced-learn.

---


Feel free to explore and experiment with the code to further enhance the model's performance or adapt it to your specific use case. If you encounter any issues or have any questions, please don't hesitate to reach out.
