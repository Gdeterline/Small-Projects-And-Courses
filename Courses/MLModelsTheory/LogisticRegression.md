# **Theoretical Course on Logistic Regression** 

## **1. Introduction to Logistic Regression** 
Logistic regression is a supervised learning algorithm used for **classification** tasks. Unlike linear regression, which predicts continuous values, logistic regression estimates the probability that an instance belongs to a particular class. It is commonly used for **binary classification** (e.g., spam vs. non-spam emails, disease vs. no disease). 

### **1.1 Why Not Use Linear Regression for Classification?** 
If we use linear regression for classification, the predicted values ($ y $) can be **greater than 1 or less than 0**, which is not valid for probabilities. Logistic regression solves this by using the **sigmoid function** to constrain the output between 0 and 1. 

---

# **2. Logistic Regression Model** 
## **2.1 The Logistic Function (Sigmoid Function)** 
The logistic function maps any real-valued number to the range (0,1), making it ideal for probability estimation:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

where $ z $ is the linear combination of input features:

$$
z = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n
$$

Thus, the logistic regression model predicts:

$$
P(y=1|x) = \frac{1}{1 + e^{-(w_0 + w_1x_1 + \dots + w_nx_n)}}
$$

where:
- $ P(y=1|x) $ is the probability that the instance belongs to class 1. 
- $ w_0, w_1, ..., w_n $ are the model parameters (to be learned). 

### **2.2 Decision Boundary** 
Since logistic regression outputs probabilities, we define a threshold to classify an observation: 

$$
y =
\begin{cases}
1, & P(y=1|x) \geq 0.5 \\
0, & P(y=1|x) < 0.5
\end{cases}
$$

The decision boundary is a **hyperplane** separating two classes.

---

# **3. Finding the Optimal Parameters** 
## **3.1 Loss Function: Log-Loss (Cross-Entropy)** 
In logistic regression, we cannot use Mean Squared Error (MSE) because it leads to non-convex optimization. Instead, we use the **log-likelihood function**, which measures how well the model predicts the true class labels:

$$
J(w) = -\frac{1}{m} \sum_{i=1}^{m} \left[y_i \log P(y_i|x_i) + (1 - y_i) \log (1 - P(y_i|x_i)) \right]
$$

where:
- $ m $ is the number of training examples.
- $ y_i $ is the actual class (0 or 1).
- $ P(y_i|x_i) $ is the predicted probability.

The goal is to **minimize this loss function**.

## **3.2 Optimization: Gradient Descent** 
To minimize $ J(w) $, we use **gradient descent**, updating parameters iteratively:

$$
w_j := w_j - \alpha \frac{\partial J}{\partial w_j}
$$

where $ \alpha $ is the learning rate, and:

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (P(y_i|x_i) - y_i) x_{ij}
$$

Gradient descent stops when changes in $ w_j $ are small, indicating convergence.

---

# **4. Assumptions of Logistic Regression** 
1. **Linearity in Log-Odds** – The relationship between input features and the log-odds of the dependent variable is linear. 
2. **Independent Observations** – Data points should not be correlated. 
3. **No Multicollinearity** – Features should not be highly correlated. 
4. **Large Sample Size** – Logistic regression performs best with large datasets. 

---

# **5. Model Evaluation Metrics** 
Since logistic regression is a classification model, we evaluate it using: 

### **5.1 Accuracy** 
$$
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
$$

However, accuracy is misleading for imbalanced datasets.

### **5.2 Precision, Recall, and F1-Score** 
- **Precision** ($ P $) – Measures how many predicted positives are actually positive:

 $$
 P = \frac{\text{True Positives}}{\text{True Positives + False Positives}}
 $$

- **Recall** ($ R $) – Measures how many actual positives were correctly identified:

 $$
 R = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}
 $$

- **F1-Score** – Harmonic mean of precision and recall:

 $$
 F1 = 2 \times \frac{P \times R}{P + R}
 $$

### **5.3 ROC Curve & AUC** 
- The **Receiver Operating Characteristic (ROC) curve** plots True Positive Rate (TPR) vs. False Positive Rate (FPR). 
- **Area Under the Curve (AUC)** measures overall model performance (higher is better). 

---

# **6. Extensions of Logistic Regression** 
### **6.1 Regularization: L1 and L2 Penalties** 
To prevent overfitting: 
- **Lasso Regression (L1)**: Adds $ \lambda \sum |w_i| $, forcing some weights to zero (feature selection). 
- **Ridge Regression (L2)**: Adds $ \lambda \sum w_i^2 $, shrinking weights but keeping all features.

### **6.2 Multinomial Logistic Regression (Softmax Regression)** 
When there are **more than two classes**, logistic regression extends to **multinomial logistic regression**, using the **softmax function**:

$$
P(y=k|x) = \frac{e^{w_k \cdot x}}{\sum_{j=1}^{K} e^{w_j \cdot x}}
$$

where $ K $ is the number of classes.

---

# **7. Summary** 
| Concept | Description |
|---------|------------|
| **Purpose** | Binary classification |
| **Model** | Uses sigmoid function to predict probabilities |
| **Loss Function** | Log-likelihood (Cross-Entropy Loss) |
| **Optimization** | Gradient Descent |
| **Assumptions** | Linearity in log-odds, independent observations |
| **Evaluation** | Accuracy, Precision, Recall, F1-Score, ROC-AUC |

---
