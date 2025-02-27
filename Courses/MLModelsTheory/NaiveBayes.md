# **Theoretical Course on Naïve Bayes** 

## **1. Introduction to Naïve Bayes** 
Naïve Bayes is a **probabilistic classification algorithm** based on **Bayes' theorem**. It is widely used for **text classification (spam filtering, sentiment analysis)** and other categorical data tasks. 

The term **"naïve"** comes from the assumption that **all features are independent**, which simplifies calculations but is often unrealistic. Despite this, Naïve Bayes performs well in many practical scenarios. 

---

# **2. Bayes' Theorem** 
Bayes' theorem describes how to update our beliefs based on new evidence:

$$
P(C | X) = \frac{P(X | C) P(C)}{P(X)}
$$

where: 
- $ P(C | X) $ is the **posterior probability** (probability of class $ C $ given feature $ X $). 
- $ P(X | C) $ is the **likelihood** (probability of feature $ X $ occurring given class $ C $). 
- $ P(C) $ is the **prior probability** (overall probability of class $ C $). 
- $ P(X) $ is the **evidence** (probability of feature $ X $ occurring, summed over all classes). 

### **2.1 Naïve Assumption (Feature Independence)** 
Naïve Bayes assumes that **features are conditionally independent**, meaning:

$$
P(X_1, X_2, \dots, X_n | C) = P(X_1 | C) P(X_2 | C) \dots P(X_n | C)
$$

This simplifies the Bayes formula to:

$$
P(C | X_1, X_2, \dots, X_n) = \frac{P(C) \prod_{i=1}^{n} P(X_i | C)}{P(X_1, X_2, ..., X_n)}
$$

Since the denominator $ P(X_1, X_2, ..., X_n) $ is constant for all classes, we only need to compute:

$$
P(C) \prod_{i=1}^{n} P(X_i | C)
$$

and choose the class with the **highest probability**.

---

# **3. Types of Naïve Bayes Classifiers** 
### **3.1 Gaussian Naïve Bayes (GNB)**
Used when features are **continuous** and assumes they follow a normal (Gaussian) distribution:

$$
P(X_i | C) = \frac{1}{\sqrt{2\pi \sigma^2_C}} e^{-\frac{(X_i - \mu_C)^2}{2\sigma_C^2}}
$$

where $ \mu_C $ and $ \sigma_C^2 $ are the **mean** and **variance** of feature $ X_i $ for class $ C $.

### **3.2 Multinomial Naïve Bayes** 
Used for **text classification** (e.g., spam detection), where features represent word counts or term frequencies. The probability is modeled using:

$$
P(X_i | C) = \frac{\text{count}(X_i, C)}{\sum_{j} \text{count}(X_j, C)}
$$

where **count($ X_i, C $)** is the number of times word $ X_i $ appears in class $ C $.

### **3.3 Bernoulli Naïve Bayes** 
Used for **binary features** (e.g., word presence/absence). The likelihood is:

$$
P(X_i | C) =
\begin{cases}
P(X_i = 1 | C) & \text{if } X_i = 1 \\
1 - P(X_i = 1 | C) & \text{if } X_i = 0
\end{cases}
$$

---

# **4. Training and Classification with Naïve Bayes** 
### **4.1 Training Phase** 
1. Compute prior probabilities $ P(C) $:

 $$
 P(C) = \frac{\text{number of instances in class } C}{\text{total number of instances}}
 $$

2. Compute conditional probabilities $ P(X_i | C) $ for each feature. 
 - For Gaussian Naïve Bayes, estimate $ \mu_C $ and $ \sigma_C^2 $ for each feature.
 - For Multinomial/Bernoulli Naïve Bayes, count word occurrences.

### **4.2 Classification Phase** 
For a new instance $ X $, compute:

$$
P(C | X) = P(C) \prod_{i=1}^{n} P(X_i | C)
$$

Select the class **$ C $ with the highest probability**.

---

# **5. Assumptions of Naïve Bayes** 
1. **Conditional Independence** – Features are assumed to be independent given the class. 
2. **Feature Relevance** – All features contribute equally to the classification. 
3. **Data Sufficient for Probability Estimation** – Enough training samples are needed to estimate probabilities accurately.

---

# **6. Model Evaluation Metrics** 
Naïve Bayes is evaluated like other classification models using:

### **6.1 Accuracy** 
$$
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
$$

### **6.2 Precision, Recall, and F1-Score** 
- **Precision** – Measures how many predicted positives are actually positive.
- **Recall** – Measures how many actual positives were correctly identified.
- **F1-Score** – Balances precision and recall.

### **6.3 Confusion Matrix** 
| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### **6.4 ROC Curve & AUC** 
- The **ROC curve** plots True Positive Rate vs. False Positive Rate. 
- **AUC (Area Under Curve)** measures overall performance.

---

# **7. Advantages and Disadvantages of Naïve Bayes** 
### **7.1 Advantages** 
✅ **Fast and Scalable** – Works well with large datasets. 
✅ **Handles Missing Data** – Uses probabilities, so missing values don't impact performance heavily. 
✅ **Performs Well for Text Classification** – Used in spam filtering, sentiment analysis, etc. 
✅ **Works Well with Small Data** – Requires less training data compared to deep learning models.

### **7.2 Disadvantages** 
❌ **Feature Independence Assumption** – Rarely holds in real-world data. 
❌ **Poor Performance with Correlated Features** – If features are highly correlated, Naïve Bayes performs poorly. 
❌ **Zero Probability Problem** – If a word never appears in training data for a class, it gets a zero probability. (Solved using **Laplace Smoothing**).

---

# **8. Laplace Smoothing (To Handle Zero Probabilities)** 
To prevent zero probabilities for unseen words, we use **Laplace Smoothing**:

$$
P(X_i | C) = \frac{\text{count}(X_i, C) + 1}{\sum_{j} \text{count}(X_j, C) + |V|}
$$

where $ |V| $ is the vocabulary size.

---

# **9. Summary** 
| Concept | Description |
|---------|------------|
| **Purpose** | Probabilistic classification |
| **Model** | Uses Bayes’ theorem with independence assumption |
| **Loss Function** | Likelihood maximization |
| **Optimization** | Closed-form solution (no gradient descent) |
| **Assumptions** | Feature independence, feature relevance |
| **Evaluation** | Accuracy, Precision, Recall, F1-Score, ROC-AUC |

---