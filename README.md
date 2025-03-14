# Bank Telemarketing Success Prediction

## Project Overview
This project compares Support Vector Machine (SVM) and Multilayer Perceptron (MLP) models for predicting customer subscription to bank term deposits. The goal is to help banking institutions improve their telemarketing campaigns by identifying potential customers more likely to subscribe to a term deposit product.

## Problem Statement
Bank telemarketing campaigns involve direct phone calls to customers to sell products or services. The effectiveness of these campaigns depends on accurately targeting potential customers who have a higher probability of subscribing. This project aims to develop and compare machine learning models that can predict customer responses to telemarketing efforts.

## Dataset
The dataset used is the "Bank Marketing" dataset from the UCI Machine Learning Repository. This data was collected from a Portuguese banking institution during a direct marketing campaign.

- **Dataset Size**: 41,188 records with 21 attributes
- **Features**: 20 predictive variables including age, job, marital status, education, housing loan status, personal loan status, etc.
- **Target Variable**: Whether the client subscribed to a term deposit (binary: 'yes' or 'no')
- **Class Distribution**: Highly imbalanced - 88% 'no' (36,548 instances) and 12% 'yes' (4,640 instances)

## Key Challenges
- Highly imbalanced dataset (88:12 ratio)
- Need for proper feature engineering and preprocessing
- Model selection for optimal performance on the minority class

## Methodology

### Data Preprocessing
1. **Label Encoding**: Converting categorical features to numerical format
2. **Feature Selection**: Removing irrelevant attributes like 'day_of_week', 'duration', 'campaign', 'pdays', and 'previous'
3. **Data Splitting**: 80% training and 20% testing with stratification to maintain class distribution
4. **Normalization**: Min-Max scaling to ensure data uniformity
5. **Handling Class Imbalance**:
   - SMOTE (Synthetic Minority Over-sampling Technique) for oversampling the minority class
   - Random undersampling of the majority class
   - Combination of SMOTE and undersampling

### Models Implemented
1. **Support Vector Machine (SVM)**
   - Hyperparameters optimized: kernel function, regularization parameter C, degrees of function, gamma values
   
2. **Multilayer Perceptron (MLP)**
   - Hyperparameters optimized: number of neurons in hidden layers, activation function, learning rate, optimizer momentum, hidden size, weight decay

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC curves and AUC
- Confusion matrices

## Results

| Metric | SVM (Test) | MLP (Test) |
|--------|------------|------------|
| Training Accuracy | 0.7327 | 0.7098 |
| Test Accuracy | 0.8319 | 0.8549 |
| Precision ("Yes") | 0.35 | 0.38 |
| Precision ("No") | 0.94 | 0.93 |
| Recall ("Yes") | 0.57 | 0.44 |
| Recall ("No") | 0.87 | 0.91 |
| F1-score ("Yes") | 0.43 | 0.41 |
| F1-score ("No") | 0.90 | 0.92 |
| AUC | 0.76 | 0.73 |

### Key Findings
- MLP achieved higher overall accuracy (85.49%) compared to SVM (83.19%)
- MLP showed better precision for the minority ("yes") class
- SVM demonstrated higher recall for both classes, capturing more positive instances
- The best results were obtained using a combination of SMOTE and undersampling
- Both models showed comparable AUC values (0.76 for SVM and 0.73 for MLP)

## Implementation Details

### Tech Stack
- Python 3.9.12
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - seaborn
  - PyTorch
  - imbalanced-learn (for SMOTE)

### Model Training
- SVM implementation using sklearn.svm.SVC
- MLP implementation using PyTorch (torch.nn)
- Hyperparameter optimization with GridSearchCV
- Adam optimizer for MLP models

## Future Work
1. Try different training techniques like boosting
2. Implement feature extraction methods like PCA
3. Explore ensemble methods to improve performance
4. Investigate other advanced sampling techniques
5. Optimize model parameters further for better recall of the minority class

## Conclusion
This project demonstrates the effectiveness of SVM and MLP models in predicting bank telemarketing campaign outcomes. While MLP shows better overall accuracy, SVM provides better recall for the minority class, which is crucial in this application. The results highlight the importance of addressing class imbalance through appropriate sampling techniques for effective predictive modelling in marketing campaigns.
