import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# Read dictionary.txt file
with open('dictionary.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()

# Extract rows with "VERB" or "INFN"
verb_inf_data = []
for i, row in enumerate(data):
    if "VERB" in row or "INFN" in row:
        verb_inf_data.append(row.strip().split("\t"))

# Create pandas dataframe
df = pd.DataFrame(verb_inf_data[1:], columns=["verb", "attributes"])

# Gathering just an even 40k of the ~41k rows to make dealing with them a bit easier
df = df.sample(frac = 1).head(40000)

# Split attributes column into separate columns
df_attributes = df['attributes'].apply(lambda x: pd.Series(x.split(',')))
df_attributes.columns = ['attribute{}'.format(i) for i in range(1, len(df_attributes.columns) + 1)]

# Merge verb column with split attributes columns
df = pd.concat([df['verb'], df_attributes], axis=1)

# Fill NaN values with empty strings
df.fillna(value='', inplace=True)

# Create parallel data frame with separate columns for each attribute
parallel_df = pd.DataFrame()
for column in df_attributes.columns:
    attribute_values = df_attributes[column].unique()
    for value in attribute_values:
        parallel_df[f'{column}_{value}'] = df['verb'].where(df[column] == value, '')

# Extract the verb and attribute columns from the dataframe
verbs = df['verb']
attributes = df.iloc[:, 1:]

mlb = MultiLabelBinarizer()
attribute_features = mlb.fit_transform(attributes.values)
attribute_columns = mlb.classes_

X_train, X_test, y_train, y_test = train_test_split(verbs, attribute_features, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = RandomForestClassifier(max_depth=None, random_state=0, n_jobs=-1, class_weight="balanced_subsample")
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)
print(classification_report(y_test, y_pred))

def predict_attributes(verb):
    verb_vectorized = vectorizer.transform([verb])
    predicted_attributes = model.predict(verb_vectorized)
    return mlb.inverse_transform(predicted_attributes)[0]

while True:
    input_verb = [str(input("Enter a verb: "))]
    print(predict_attributes(str(input_verb)))