from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

np.random.seed(0)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())  # show the first 5 rows

# Add a column with the ground-truth category for each flower
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print(df.head())

# Keep ~75% of the data as the training set
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
print(df.head())

# Split the data into training and testing sets
train, test = df[df['is_train'] == True], df[df['is_train'] == False]

print(len(train))
print(len(test))

features = df.columns[0:4]
print(features)

# Convert each species name into digits
y = pd.factorize(train['species'])[0]
print(y)

# Train a random forest with 10 trees
clf = RandomForestClassifier(n_estimators=10, random_state=0)
clf.fit(train[features], y)

result = clf.predict(test[features])
#
print(result)

# Show the predicted probabilities for all the test samples
proba = clf.predict_proba(test[features])
print(proba)

# Show the corresponding predicted species
preds = iris.target_names[clf.predict(test[features])]
print(preds)

# Show the ground-truth species
print(test['species'].head())

conf_matrix = pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])
print(conf_matrix)
