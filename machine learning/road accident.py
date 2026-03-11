import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("Road accident 2020 india.csv")
df.head()
df.shape
df.info()
df.isnull().sum()
df = df.dropna()
df.columns
plt.figure(figsize=(14,10))

# 1️⃣ Outcome vs Count
plt.subplot(2,2,1)
sns.barplot(x='Outcome of Incident', y='Count', data=df, estimator=sum)
plt.title("Outcome of Incident vs Count")
plt.xticks(rotation=30)

# 2️⃣ Cause Category vs Count
plt.subplot(2,2,2)
sns.barplot(x='Cause category', y='Count', data=df, estimator=sum)
plt.title("Cause Category vs Count")
plt.xticks(rotation=45)

# 3️⃣ Top Cities vs Count
plt.subplot(2,2,3)
top_cities = df.groupby('Million Plus Cities')['Count'].sum().sort_values(ascending=False).head(5)
sns.barplot(x=top_cities.index, y=top_cities.values)
plt.title("Top 5 Cities by Accident Count")
plt.xticks(rotation=30)

# 4️⃣ Cause vs Outcome
plt.subplot(2,2,4)
sns.barplot(x='Cause category', y='Count', hue='Outcome of Incident',
            data=df, estimator=sum)
plt.title("Cause vs Outcome Analysis")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



le_city = LabelEncoder()
le_cause = LabelEncoder()
le_outcome = LabelEncoder()

df['City_encoded'] = le_city.fit_transform(df['Million Plus Cities'])
df['Cause_encoded'] = le_cause.fit_transform(df['Cause category'])
df['Outcome_encoded'] = le_outcome.fit_transform(df['Outcome of Incident'])

X = df[['City_encoded', 'Cause_encoded', 'Outcome_encoded']]
y = df['Count']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Count")
plt.ylabel("Predicted Count")
plt.title("Actual vs Predicted Accident Count")
plt.show()

