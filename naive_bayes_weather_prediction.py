# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Read the CSV file
df = pd.read_csv('Playing_Tennis.csv')
print(df)

# Label Encoding for categorical data
Numerics = LabelEncoder()
print(df.columns)

# Separate inputs and target
inputs = df.drop('Play ', axis='columns')
target = df['Play ']
print(target)

# Encode the input features
inputs['Outlook_n'] = Numerics.fit_transform(inputs['Outlook'])
inputs['Temp_n'] = Numerics.fit_transform(inputs['Temp'])
inputs['Humidity_n'] = Numerics.fit_transform(inputs['Humidity'])
inputs['Windy_n'] = Numerics.fit_transform(inputs['Windy'])
print(inputs)

# Drop the original columns
inputs_n = inputs.drop(['Outlook', 'Temp', 'Humidity', 'Windy'], axis='columns')
print(inputs_n)

# Train the Naive Bayes classifier
Classifier = GaussianNB()
Classifier.fit(inputs_n, target)

# Calculate and display accuracy
accuracy = Classifier.score(inputs_n, target)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display the DataFrame
print(df)
