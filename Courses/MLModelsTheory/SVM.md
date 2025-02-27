# **Theoretical Course on Support Vector Machines (SVM)** 

## **1. Introduction to Support Vector Machines** 
Support Vector Machines (SVM) is a powerful **supervised learning algorithm** used for **classification** and **regression** tasks. It is particularly effective in **high-dimensional spaces** and when the number of dimensions is greater than the number of samples. 

SVM works by finding the **optimal decision boundary** that **maximizes the margin** between two classes. This boundary is called the **hyperplane**, and it is determined using **support vectors**, which are the data points closest to the decision boundary. 

---

# **2. The SVM Model** 
## **2.1 Decision Boundary and Hyperplane** 
A hyperplane is a decision boundary that separates classes. In an **n-dimensional space**, the hyperplane is an **(n-1)-dimensional plane** defined as:

$$
w \cdot x + b = 0
$$

where: 
- $ w $ is the weight vector (normal to the hyperplane). 
- $ x $ is the feature vector. 
- $ b $ is the bias term. 

### **2.1.1 Classification Rule** 
For a **binary classification problem**, a new data point $ x $ is classified as: 

$$
y =
\begin{cases}
+1, & w \cdot x + b \geq 0 \\
-1, & w \cdot x + b < 0
\end{cases}
$$

### **2.1.2 Margin and Support Vectors** 
- **Margin**: The distance between the hyperplane and the nearest data points from each class. 
- **Support Vectors**: The data points that lie on the margin boundaries. These points **define the optimal hyperplane**. 

---

# **3. Hard Margin vs. Soft Margin SVM** 
## **3.1 Hard Margin SVM** 
In an **ideal scenario with linearly separable data**, SVM finds the hyperplane that **maximizes the margin** without allowing any misclassified points:

$$
\min_{w, b} \frac{1}{2} ||w||^2
$$

subject to:

$$
y_i (w \cdot x_i + b) \geq 1, \quad \forall i
$$

where $ y_i $ is the class label ($ \pm 1 $) and $ x_i $ are the training samples.

### **3.1.1 Problem with Hard Margin** 
- **Sensitive to outliers**: A single outlier can significantly shift the decision boundary. 
- **Not suitable for non-linearly separable data**. 

## **3.2 Soft Margin SVM (Handling Overlap and Noise)** 
To allow misclassified points, we introduce a **slack variable** $ \xi_i $, leading to:

$$
\min_{w, b} \frac{1}{2} ||w||^2 + C \sum \xi_i
$$

subject to:

$$
y_i (w \cdot x_i + b) \geq 1 - \xi_i, \quad \forall i, \quad \xi_i \geq 0
$$

where $ C $ is a **regularization parameter** controlling the trade-off between **margin maximization** and **misclassification tolerance**.

- **High $ C $**: Low tolerance for misclassification → risk of overfitting. 
- **Low $ C $**: More tolerance for misclassification → better generalization. 

---

# **4. Kernel Trick: Handling Non-Linearly Separable Data** 
When data is **not linearly separable**, SVM transforms it into a **higher-dimensional space** where it becomes separable using the **kernel trick**.

### **4.1 Common Kernel Functions** 
1. **Linear Kernel** (for linearly separable data): 
 $$
 K(x_i, x_j) = x_i \cdot x_j
 $$
2. **Polynomial Kernel** (for polynomial decision boundaries): 
 $$
 K(x_i, x_j) = (x_i \cdot x_j + c)^d
 $$
3. **Radial Basis Function (RBF) Kernel** (most widely used, captures complex relationships): 
 $$
 K(x_i, x_j) = \exp \left(-\gamma ||x_i - x_j||^2 \right)
 $$
 - **Small $ \gamma $**: Large influence → risk of underfitting. 
 - **Large $ \gamma $**: Small influence → risk of overfitting. 
4. **Sigmoid Kernel** (similar to neural networks): 
 $$
 K(x_i, x_j) = \tanh (\alpha x_i \cdot x_j + c)
 $$

---

# **5. Dual Formulation & Lagrange Multipliers** 
Instead of solving the primal optimization problem, we use the **Lagrange dual formulation**, which allows SVM to work efficiently with kernels.

$$
\max_{\alpha} \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y_i y_j K(x_i, x_j)
$$

subject to:

$$
0 \leq \alpha_i \leq C, \quad \sum_{i=1}^{m} \alpha_i y_i = 0
$$

where $ \alpha_i $ are Lagrange multipliers. The **support vectors** are the data points where $ \alpha_i > 0 $.

---

# **6. Assumptions of SVM** 
1. **Data is separable or approximately separable** (linearly or via kernel). 
2. **Feature Scaling is important** (SVM is sensitive to large feature values). 
3. **Choice of Kernel Matters** (poor choice may lead to overfitting or underfitting). 

---

# **7. Model Evaluation Metrics** 
SVM is evaluated using standard classification metrics:

### **7.1 Accuracy** 
$$
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
$$

### **7.2 Precision, Recall, and F1-Score** 
- **Precision**: How many predicted positives are actually positive. 
- **Recall**: How many actual positives were correctly predicted. 
- **F1-Score**: Harmonic mean of precision and recall.

### **7.3 Confusion Matrix** 
| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### **7.4 ROC Curve & AUC** 
- **ROC Curve**: Plots True Positive Rate vs. False Positive Rate. 
- **AUC (Area Under Curve)**: Measures the classifier’s ability to distinguish between classes.

---

# **8. Advantages and Disadvantages of SVM** 
### **8.1 Advantages** 
✅ **Effective in high-dimensional spaces** (suitable for small datasets with many features). 
✅ **Robust to overfitting** (especially with proper regularization). 
✅ **Works well with clear margin separation**. 
✅ **Kernel trick allows handling non-linearly separable data**. 

### **8.2 Disadvantages** 
❌ **Computationally expensive** for large datasets. 
❌ **Sensitive to choice of kernel and hyperparameters**. 
❌ **Does not perform well with noisy data and overlapping classes**. 
❌ **Difficult to interpret compared to logistic regression**. 

---

# **9. Summary** 
| Concept | Description |
|---------|------------|
| **Purpose** | Classification & regression |
| **Model** | Finds optimal hyperplane maximizing margin |
| **Loss Function** | Hinge loss |
| **Optimization** | Quadratic programming (dual formulation) |
| **Kernel Trick** | Maps data to higher dimensions |
| **Evaluation** | Accuracy, Precision, Recall, F1-Score, ROC-AUC |

---