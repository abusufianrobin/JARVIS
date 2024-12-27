# Import Required Libraries
import pandas as pd  # For data handling
from sklearn.preprocessing import LabelEncoder, MinMaxScaler  # For encoding and scaling
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.naive_bayes import GaussianNB  # Naive Bayes classifier
from sklearn.metrics import accuracy_score  # For accuracy evaluation

# Load the Dataset
df = pd.read_csv('weather_history_bangladesh.csv')

# Remove Duplicate Rows
df = df.drop_duplicates()

# Check for Missing Values
print("Missing Values:\n", df.isnull().sum())

# Fill or Drop Missing Values (if any)
df = df.dropna()  # Dropping rows with missing values

# Inspect Dataset
print("Dataset Head:\n", df.head())
print("Dataset Info:\n", df.info())

# Initialize LabelEncoder
encoder = LabelEncoder()

# Separate Features and Target
inputs = df.drop('condition', axis='columns')
target = df['condition']

# Apply Label Encoding to Categorical Columns
inputs['date_n'] = encoder.fit_transform(inputs['date'])
inputs['time_n'] = encoder.fit_transform(inputs['time'])
inputs['temperature_fahrenheit_n'] = encoder.fit_transform(inputs['temperature_fahrenheit'])
inputs['dew_point_fahrenheit_n'] = encoder.fit_transform(inputs['dew_point_fahrenheit'])
inputs['humidity_percentage_n'] = encoder.fit_transform(inputs['humidity_percentage'])
inputs['wind_n'] = encoder.fit_transform(inputs['wind'])
inputs['wind_speed_mph_n'] = encoder.fit_transform(inputs['wind_speed_mph'])
inputs['wind_gust_mph_n'] = encoder.fit_transform(inputs['wind_gust_mph'])
inputs['pressure_in_n'] = encoder.fit_transform(inputs['pressure_in'])
inputs['precip._in_n'] = encoder.fit_transform(inputs['precip._in'])

# Drop Original Non-Numeric Columns
inputs_n = inputs.drop(['date', 'time', 'temperature_fahrenheit', 'dew_point_fahrenheit',
                        'humidity_percentage', 'wind', 'wind_speed_mph', 'wind_gust_mph',
                        'pressure_in', 'precip._in'], axis='columns')

# Normalize Data
scaler = MinMaxScaler()
inputs_scaled = scaler.fit_transform(inputs_n)

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(inputs_scaled, target, test_size=0.2, random_state=42)

# Initialize and Train Naive Bayes Classifier
Classifier = GaussianNB()
Classifier.fit(X_train, y_train)

# Evaluate Model on Test Data
y_pred = Classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predict New Data Example
new_data = [[0, 0, 8, 6, 38, 0, 0, 0, 19, 0]]  # Replace with actual data values
new_data_scaled = scaler.transform(new_data)
prediction = Classifier.predict(new_data_scaled)
print(f"Prediction for new data: {prediction}")
