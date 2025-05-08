import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder # type: ignore
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score # type: ignore
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from xgboost import XGBClassifier # type: ignore



data = pd.read_csv('your/path/to/the/data.csv')
data.head()
print(data.info())
print(data.describe()) # description of the data clearly
print(data.isnull().sum())  # Checking for missing values

data = data[data["disposition"] != "CANDIDATE"] # herre we are removing all the rows having the candidate as its disposition value for the binary classification we dont need that  

# Dropping the useless columns
# data = data.drop(columns=['koi_name'])
data = data.drop(['koi_name', 'false_positive_type', 'fp_not_transit', 'fp_stellar_eclipse', 'fp_centroid_offset', 'fp_contamination'], axis = 1)
data = data.dropna(thresh=int(0.7 * len(data)), axis=1) # here we will be keeping a threshold as a 30% so for every column if there are more than 30% null values then its gonna drop the column
data.info()

# here we are just converting the categorical data confirmed and the false positive as the numerical values as the 0 and 1
label_encoder = LabelEncoder()
data['disposition'] = label_encoder.fit_transform(data['disposition']) #for confirmed it is 0 and for false positive it is 1
print(data['disposition'])

# Imputeing the missing values
numeric_cols = data.select_dtypes(include=[np.number]).columns #exttracting all the columns having the numerical data
categorical_cols = data.select_dtypes(include=[object]).columns #extracting all the columns haviong categorical data
# print(numeric_cols)
# print(categorical_cols)

numeric_imputer = SimpleImputer(strategy='mean') # handling the missinig data using the mean of the column
categorical_imputer = SimpleImputer(strategy='most_frequent') # handling the missing data using the most frequently appeared onw

data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols]) #after handling the missing data
# data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols]) # modifying the dataset again
# Check for categorical columns before applying imputation
categorical_cols = data.select_dtypes(include=[object]).columns
print(f'Categorical columns: {categorical_cols}')

if len(categorical_cols) > 0:
    categorical_imputer = SimpleImputer(strategy='most_frequent') # handle missing categorical data using the most frequent value
    data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])
else:
    print("No categorical columns to impute.")


# Standardize numeric columns
scaler = StandardScaler()  # to be honest by using the standard scaler the accuracy is just improved by 0.01 and the just got some more accurate values like around 5 and very small or invisible amount of improvement in the roc-auc score
data[numeric_cols] = scaler.fit_transform(data[numeric_cols]) # standardisation the data using the standarad scaler 

data['disposition'] = data['disposition'].astype(int)


# Separate target variable
X = data.drop(columns=['disposition'])
y = data['disposition']



# Splitting the the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
data.info()
print(data.describe())

# Convert to new dataset for machine learning
processed_data = pd.concat([X_train, y_train], axis=1)
processed_data.to_csv("processed_exoplanet_data.csv", index=False) # here we can also create an csv file for the processed data like if we use another file for the regression models it will be useful but for now we are daoing all the things here only 

#-------------------------------------------------------------------------------
