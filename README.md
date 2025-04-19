# Multi-Input, Multi-Output Neural Network for Predicting Employee Attrition and Department

This project demonstrates how to build a multi-input, multi-output neural network using TensorFlow/Keras to predict both **employee attrition** (binary classification) and **department** (multi-class classification) from HR data.

---

## 📁 Dataset

- **Source**: [Attrition Dataset](https://static.bc-edx.com/ai/ail-v-1-0/m19/lms/datasets/attrition.csv)
- Features include demographic, job satisfaction, and performance-related information.

---

## 🧮 Project Structure

### Part 1: Data Preprocessing

1. **Import data and dependencies**
2. **Separate features (`X_df`) and labels (`y_df`)**
    - X features include: `Age`, `Education`, `DistanceFromHome`, `JobSatisfaction`, `OverTime`, etc.
    - y labels include: `Attrition`, `Department`
3. **One-hot encode categorical columns**
    - `"OverTime"` in `X`
    - `"Department"` and `"Attrition"` in `y`
4. **Scale numeric features** using `StandardScaler`
5. **Split the dataset** into training and testing sets

---

### Part 2: Model Architecture

The model includes:

- **Three input branches**:
  - Shared features (scaled X features)
  - One-hot encoded department input
  - One-hot encoded attrition input
- **Two output branches**:
  - `attrition_output` – binary classifier using sigmoid activation
  - `department_output` – multi-class classifier using softmax activation

Hidden layers include:
- Shared layers: 32 and 16 units (ReLU)
- Department branch: 16 units (ReLU)
- Attrition branch: 8 units (ReLU)
- Final merged layer before output: 16 units (ReLU)

---

## 🔧 Model Compilation

The model uses:
- Optimizer: `adam`
- Loss functions:
  - `binary_crossentropy` for attrition
  - `categorical_crossentropy` for department
- Metrics:
  - `accuracy` for both outputs

---

## 🧪 Training

```python
history = model.fit(
    x=[X_train_scaled, department_train_df, attrition_train_df],
    y={
        'attrition_output': y_train_binary,
        'department_output': department_train_df
    },
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

