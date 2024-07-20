import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


file_path = 'IRIS.csv'
iris_df = pd.read_csv(file_path)


print("First few rows of the dataset:")
print(iris_df.head())

X = iris_df.drop(columns=['species'])
y = iris_df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the Random Forest Classifier
clf = RandomForestClassifier(random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred, target_names=iris_df['species'].unique())

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
