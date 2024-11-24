import pandas as pd
from sklearn.impute import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the data
data = pd.read_csv("F:\\FALL SEMESTER 2024\\Digital Tools for Finance\\Project\\Processes Data\\Gross pay.csv")

# Handle categorical variables
le = LabelEncoder()
data['Job Role'] = le.fit_transform(data['Job Role'])
# ... similar for other categorical features

# Normalize numerical features
scaler = StandardScaler()
data[['Gross Pay Men', 'Gross Pay Women', 'Pay Gap Percentage']] = scaler.fit_transform(data[['Gross Pay Men', 'Gross Pay Women', 'Pay Gap Percentage']])

# Select features and target variable
X = data.drop(['Gross Pay Men', 'Gross Pay Women', 'Pay Gap Percentage'], axis=1)
y = data[['Gross Pay Men', 'Gross Pay Women', 'Pay Gap Percentage']]

# Create a KNN regressor
knn_regressor = KNeighborsRegressor(n_neighbors=5)

# Fit the model
knn_regressor.fit(X, y)

# Predict missing values
data_with_missing_values = ... # Load data with missing values
predicted_values = knn_regressor.predict(data_with_missing_values)

# Fill missing values in the original data
data_with_missing_values.fillna(predicted_values, inplace=True)

# Inverse transform the scaled values
data_with_missing_values[['Gross Pay Men', 'Gross Pay Women', 'Pay Gap Percentage']] = scaler.inverse_transform(data_with_missing_values[['Gross Pay Men', 'Gross Pay Women', 'Pay Gap Percentage']])
# Identify missing values
print(data.isnull().sum())

# Impute missing values with mean imputation
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Create a new DataFrame with imputed values
data_imputed = pd.DataFrame(data_imputed, index=data.index, columns=data.columns)

# Analyze the imputed data
print(data_imputed.head())