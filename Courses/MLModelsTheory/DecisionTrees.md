# **Theoretical Course on Decision Trees** 

## **1. Introduction to Decision Trees** 
Decision Trees are **supervised learning algorithms** used for **classification and regression**. They model decisions in a **tree-like structure**, where each internal node represents a **decision rule**, each branch represents an **outcome**, and each leaf node represents a **class label (classification)** or a **continuous value (regression)**.

Decision Trees are intuitive, interpretable, and can handle both numerical and categorical data.

---

# **2. Structure of a Decision Tree** 
A Decision Tree consists of: 
- **Root Node**: The starting point containing the entire dataset. 
- **Internal Nodes**: Points where the data splits based on feature conditions. 
- **Branches**: Outcomes of decision rules leading to child nodes. 
- **Leaf Nodes**: The final output (class label or regression value). 

### **Example of a Simple Decision Tree (for Classification)** 
```
 Is Age > 30?
 / \
 Yes No
 / \
 Has Job? Student?
 / \ / \
 Yes No Yes No
 / \ / \
 Approve Reject Reject Approve
```
This tree classifies loan applications based on age, employment status, and student status.

---

# **3. How Decision Trees Work** 

## **3.1 Splitting Criteria: Selecting the Best Feature** 
To build an optimal tree, we need to decide **which feature to split on at each step**. This is done using **impurity measures**:

### **3.1.1 Entropy & Information Gain (for Classification Trees)**
Entropy measures **uncertainty** in a dataset:

$$
H(S) = -\sum p_i \log_2(p_i)
$$

where $ p_i $ is the proportion of class $ i $. 

**Information Gain (IG)** measures the reduction in entropy after a split:

$$
IG(S, A) = H(S) - \sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} H(S_v)
$$

**Higher Information Gain → Better Split.**

### **3.1.2 Gini Impurity (for Classification Trees)**
Gini impurity is another measure of uncertainty:

$$
Gini(S) = 1 - \sum p_i^2
$$

A split with **lower Gini impurity** is preferred.

### **3.1.3 Mean Squared Error (for Regression Trees)**
For regression, we minimize the Mean Squared Error (MSE):

$$
MSE = \frac{1}{n} \sum (y_i - \hat{y})^2
$$

where $ y_i $ are actual values and $ \hat{y} $ is the predicted mean.

---

# **4. Building a Decision Tree** 
### **Step 1: Select the Best Feature** 
- Compute **Information Gain** (or Gini) for each feature.
- Choose the **feature with the highest Information Gain**.

### **Step 2: Split the Dataset** 
- Divide data based on the chosen feature.
- Assign subsets to child nodes.

### **Step 3: Repeat Recursively** 
- Repeat steps 1 and 2 until we meet a **stopping criterion**:
 - **All instances belong to the same class**.
 - **No more features to split**.
 - **Tree reaches maximum depth**.

---

# **5. Overfitting & Regularization** 
A Decision Tree can **overfit** if it grows too deep, capturing noise instead of general patterns. To prevent this, we use:

### **5.1 Pruning** 
Pruning removes unnecessary branches. 
- **Pre-Pruning (Early Stopping)**: Stop splitting when the tree reaches a certain depth or has too few samples. 
- **Post-Pruning**: Grow the full tree, then remove weak branches.

### **5.2 Minimum Split Size** 
Set a minimum number of samples required to split a node.

### **5.3 Maximum Tree Depth** 
Limit the depth to prevent excessive complexity.

---

# **6. Decision Tree Variants** 
### **6.1 Classification Trees (CART)**
- Output: Discrete class labels.
- Uses **Gini impurity** or **entropy** for splitting.

### **6.2 Regression Trees (CART)**
- Output: Continuous numerical values.
- Uses **Mean Squared Error (MSE)** for splitting.

### **6.3 ID3 (Iterative Dichotomiser 3)**
- Uses **Information Gain** as the splitting criterion.
- Works only with **categorical features**.

### **6.4 C4.5 (Improvement of ID3)**
- Handles **both categorical and numerical features**.
- Uses **Gain Ratio** to penalize too many splits.

---

# **7. Assumptions of Decision Trees** 
1. **Features are independent** (though correlations may exist). 
2. **Data is clean (without too many missing values)**. 
3. **Class distributions are somewhat balanced** (otherwise, pruning or boosting might be needed). 

---

# **8. Model Evaluation Metrics** 
Decision Trees use standard classification and regression metrics:

### **8.1 Classification Metrics** 
- **Accuracy**: Correct predictions out of total predictions.
- **Precision, Recall, F1-Score**: Evaluate performance for imbalanced classes.
- **Confusion Matrix**: Shows correct vs. incorrect classifications.

### **8.2 Regression Metrics** 
- **Mean Squared Error (MSE)**: Measures average squared error.
- **Mean Absolute Error (MAE)**: Measures absolute error.
- **R² Score**: Measures goodness of fit.

---

# **9. Advantages & Disadvantages** 
### **9.1 Advantages** 
✅ **Easy to understand and interpret**. 
✅ **Handles both categorical and numerical data**. 
✅ **Requires minimal data preprocessing** (no need for feature scaling). 
✅ **Can model complex relationships** (via deep trees). 

### **9.2 Disadvantages** 
❌ **Prone to overfitting** (without pruning). 
❌ **Sensitive to noisy data** (small changes can alter the tree structure). 
❌ **Not ideal for smooth decision boundaries** (unlike SVM). 
❌ **Computationally expensive** for large datasets. 

---

# **10. Summary** 
| Concept | Description |
|---------|------------|
| **Purpose** | Classification & Regression |
| **Splitting Criteria** | Entropy, Gini, MSE |
| **Loss Function** | Entropy, Gini for classification; MSE for regression |
| **Overfitting Prevention** | Pruning, depth control, minimum split size |
| **Evaluation** | Accuracy, F1-score (classification), MSE (regression) |

---