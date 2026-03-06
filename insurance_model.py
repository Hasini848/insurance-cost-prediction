import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df = pd.read_csv("insurance.csv")

# BMI Category
def bmi_category(bmi):
    if bmi < 18.5:
        return 0
    elif bmi < 25:
        return 1
    elif bmi < 30:
        return 2
    else:
        return 3

df["bmi_category"] = df["bmi"].apply(bmi_category)

# Age Group
def age_group(age):
    if age < 30:
        return 0
    elif age < 50:
        return 1
    else:
        return 2

df["age_group"] = df["age"].apply(age_group)

# Interaction feature
df["smoker"] = df["smoker"].map({"yes":1,"no":0})
df["smoker_bmi"] = df["smoker"] * df["bmi"]

# Outlier removal (IQR)
Q1 = df["charges"].quantile(0.25)
Q3 = df["charges"].quantile(0.75)

IQR = Q3-Q1

df = df[(df["charges"] >= Q1-1.5*IQR) & (df["charges"] <= Q3+1.5*IQR)]

# Features
X = df[["age","bmi","children","smoker_bmi"]]
y = df["charges"]

# Train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test,pred)
rmse = np.sqrt(mean_squared_error(y_test,pred))
r2 = r2_score(y_test,pred)

print("MAE:",mae)
print("RMSE:",rmse)
print("R2 Score:",r2)

joblib.dump(model,"insurance_model.pkl")