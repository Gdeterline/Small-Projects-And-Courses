# **Data Science Cheat Sheet** 

Cheat sheet covers essential functions from **Exploratory Data Analysis (EDA)** to **Model Deployment**, using **pandas, matplotlib, seaborn, scikit-learn, and PyTorch**.  

---

## **Data Handling (pandas)**
### **Load & Inspect Data**
```python
import pandas as pd

df = pd.read_csv("data.csv")  # Load CSV
df.head()                     # First 5 rows
df.info()                     # Data types, nulls
df.describe()                 # Summary stats
df.columns                    # List columns
df.nunique()                  # Count unique values
df.duplicated().sum()         # Check duplicates
df.isnull().sum()             # Count missing values
```

### **Missing Values**
```python
df.fillna(df.mean(), inplace=True)  # Fill with mean
df.dropna(inplace=True)             # Drop missing values
```

### **Filtering & Selecting**
```python
df[df['col1'] > 50]             # Filter rows
df[['col1', 'col2']]            # Select columns
df.sort_values('col1', ascending=False)  # Sort
```

---

## **Data Visualization (matplotlib & seaborn)**
```python
import matplotlib.pyplot as plt
import seaborn as sns
```

### **Basic Plotting**
```python
df['col1'].hist()                    # Histogram
df.plot(kind='box')                   # Boxplot
df.plot(kind='scatter', x='col1', y='col2')  # Scatter plot
plt.show()
```

### **Seaborn for Better Visuals**
```python
sns.histplot(df['col1'], kde=True)    # Histogram with KDE
sns.boxplot(x='category', y='value', data=df)  # Boxplot
sns.pairplot(df)                      # Pairplot for relationships
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")  # Correlation heatmap
plt.show()
```

---

## **Feature Engineering**

### **Encoding Categorical Variables**

Mapping Categorical to Numerical
```python
df['category'] = df['category'].map({'A': 0, 'B': 1, 'C': 2})
```

One-Hot Encoding & Label Encoding
```python
df = pd.get_dummies(df, drop_first=True)  # One-Hot Encoding
from sklearn.preprocessing import LabelEncoder
df['category'] = LabelEncoder().fit_transform(df['category'])  # Label Encoding
```

### **Scaling Features**
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()  # Normalize to mean=0, std=1
df_scaled = scaler.fit_transform(df[['col1', 'col2']])

minmax_scaler = MinMaxScaler()  # Scale between 0 and 1
df_scaled = minmax_scaler.fit_transform(df[['col1', 'col2']])
```

---

## **Splitting Data**
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## **Model Training (scikit-learn)**
### **Linear Regression**
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### **Logistic Regression**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### **Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## **Model Evaluation**
### **Regression Metrics**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

### **Classification Metrics**
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### **ROC Curve**
```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

---

## ** Hyperparameter Tuning**
### **Grid Search**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
```

### **Random Search**
```python
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(RandomForestClassifier(), param_grid, n_iter=5, cv=5)
random_search.fit(X_train, y_train)

print(random_search.best_params_)
```

---

## **Deep Learning with PyTorch**
```python
import torch
import torch.nn as nn
import torch.optim as optim
```

### **Creating a Simple Neural Network**
```python
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = NeuralNet(input_size=X_train.shape[1])
criterion = nn.MSELoss()  # Loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

### **Training the Model**
```python
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/100], Loss: {loss.item()}')
```

---

## **Model Deployment**
### **Save Model**
```python
import joblib
joblib.dump(model, 'model.pkl')
```

### **Load Model**
```python
model = joblib.load('model.pkl')
```

### **Deploy with Flask**
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
```

---

## **Summary Table**
| Step | Library | Key Functions |
|------|---------|--------------|
| **EDA** | pandas | `df.info()`, `df.describe()`, `df.isnull().sum()` |
| **Visualization** | seaborn, matplotlib | `sns.heatmap()`, `plt.scatter()` |
| **Feature Engineering** | pandas, scikit-learn | `LabelEncoder()`, `StandardScaler()` |
| **Model Training** | scikit-learn | `model.fit()`, `model.predict()` |
| **Evaluation** | scikit-learn | `accuracy_score()`, `confusion_matrix()` |
| **Tuning** | scikit-learn | `GridSearchCV()`, `RandomizedSearchCV()` |
| **Deep Learning** | PyTorch | `nn.Linear()`, `optim.Adam()` |
| **Deployment** | joblib, Flask | `joblib.dump()`, `Flask()` |

---