import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('data/traffic_data.csv')

# Feature Engineering
data['Total_Count'] = data['Vehicle_Count'] + data['Pedestrian_Count']

# Splitting Data
X = data[['Hour', 'Total_Count']]
y = data['Signal_Duration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed data
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)
