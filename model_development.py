import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -------------------------------
# 1. Load Dataset
# -------------------------------

df = pd.read_csv("train.csv")   # Make sure train.csv is in same folder

# -------------------------------
# 2. Feature Selection
# -------------------------------

features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
target = "Survived"

X = df[features]
y = df[target]

# -------------------------------
# 3. Handle Missing Values
# -------------------------------

X["Age"] = X["Age"].fillna(X["Age"].median())
X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

# -------------------------------
# 4. Preprocessing
# -------------------------------

numeric_features = ["Age", "Fare"]
categorical_features = ["Pclass", "Sex", "Embarked"]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# -------------------------------
# 5. Build Model
# -------------------------------

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# -------------------------------
# 6. Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 7. Train Model
# -------------------------------

model.fit(X_train, y_train)

# -------------------------------
# 8. Evaluation
# -------------------------------

y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# 9. Save Model
# -------------------------------

joblib.dump(model, "titanic_survival_model.pkl")
print("\nModel saved as titanic_survival_model.pkl")
