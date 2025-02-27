# **Theoretical Course on Linear Regression** 

## **1. Introduction to Linear Regression** 
Linear regression is a fundamental supervised learning algorithm used for modeling the relationship between a dependent variable $ y $ and one or more independent variables $ x $. The objective is to find a **best-fit line** that minimizes the difference between predicted and actual values. 

---

# **2. The Equation of a Linear Model** 
For **simple linear regression**, the model is defined as: 

$$
y = w_0 + w_1 x + \epsilon
$$ 

where: 
- $ y $ is the dependent variable (target/output). 
- $ x $ is the independent variable (feature/input). 
- $ w_0 $ is the **intercept** (value of $ y $ when $ x = 0 $). 
- $ w_1 $ is the **slope** (how much $ y $ changes when $ x $ increases by 1 unit). 
- $ \epsilon $ is the **error term**, accounting for randomness or unobserved factors. 

For **multiple linear regression**, the equation extends to: 
$$
y = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n + \epsilon
$$ 
where $ n $ is the number of features.

---

# **3. Finding the Best-Fit Line** 
The best-fit line is the one that minimizes the total error. This is achieved by minimizing a loss function, typically the **Mean Squared Error (MSE)**.

## **3.1 Cost Function (Mean Squared Error)** 
The cost function measures how well a given line fits the data. The most commonly used cost function for linear regression is:

$$
J(w_0, w_1) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$

where: 
- $ m $ is the number of training examples. 
- $ y_i $ is the actual value of the target variable. 
- $ \hat{y}_i $ is the predicted value from the model. 

The goal is to find $ w_0 $ and $ w_1 $ that **minimize** $ J(w_0, w_1) $.

---

# **4. Optimization: Methods to Minimize the Cost Function** 
There are two main methods to find the best parameters $ w_0, w_1 $: 

## **4.1 Analytical Solution: Normal Equation** 
For small datasets, we can compute the optimal parameters **directly** using linear algebra:

$$
W = (X^T X)^{-1} X^T y
$$

where: 
- $ X $ is the matrix of input features (with a column of ones for $ w_0 $). 
- $ y $ is the vector of target values. 
- $ W $ is the vector of coefficients ($ w_0, w_1, ..., w_n $). 

This method works well for small datasets but becomes computationally expensive for large datasets due to matrix inversion.

## **4.2 Gradient Descent: Iterative Optimization** 
For large datasets, **gradient descent** is preferred. It updates $ w_0 $ and $ w_1 $ iteratively to minimize the cost function.

### **4.2.1 Gradient Descent Formula** 
1. Compute the gradient (partial derivative of cost function):

 $$
 \frac{\partial J}{\partial w_0} = -\frac{2}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)
 $$

 $$
 \frac{\partial J}{\partial w_1} = -\frac{2}{m} \sum_{i=1}^{m} x_i (y_i - \hat{y}_i)
 $$

2. Update the parameters:

 $$
 w_0 := w_0 - \alpha \frac{\partial J}{\partial w_0}
 $$

 $$
 w_1 := w_1 - \alpha \frac{\partial J}{\partial w_1}
 $$

where $ \alpha $ (learning rate) controls the step size. 

### **4.2.2 Convergence** 
Gradient descent stops when changes in $ w_0, w_1 $ are small, indicating the minimum cost is reached.

---

# **5. Assumptions of Linear Regression** 
For linear regression to be valid, the following assumptions must hold: 

1. **Linearity** – The relationship between $ x $ and $ y $ must be linear. 
2. **Independence of errors** – No correlation between residuals. 
3. **Homoscedasticity** – Constant variance of residuals. 
4. **Normality of errors** – Residuals should be normally distributed. 
5. **No multicollinearity** (for multiple regression) – Features should not be highly correlated.

---

# **6. Evaluating the Model** 
### **6.1 R² Score (Coefficient of Determination)** 
The **R² score** measures how well the regression model explains the variance in $ y $: 

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

- $ R^2 = 1 $ → Perfect fit. 
- $ R^2 = 0 $ → Model is no better than a horizontal line (mean of $ y $). 
- $ R^2 < 0 $ → Model performs worse than the mean.

### **6.2 Mean Squared Error (MSE)** 
$$
MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$

Lower MSE means better performance.

---

# **7. Extensions of Linear Regression** 
### **7.1 Regularized Regression** 
Regularization prevents overfitting: 
- **Ridge Regression** (L2 penalty): Adds a squared penalty term $ \lambda \sum w_i^2 $. 
- **Lasso Regression** (L1 penalty): Forces some coefficients to zero, helping with feature selection.

### **7.2 Polynomial Regression** 