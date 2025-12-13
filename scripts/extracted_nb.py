# Cell 0
import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import kruskal, ttest_ind
from scipy.stats import spearmanr
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import re

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Cell 1
df = pd.read_excel('data.xlsx', sheet_name='data')
df.head()

# Cell 4
df.info()

# Cell 5
# From the info, it seems a lot of columns have high null ratio. Check null rate in all columns
null_percent = df.isnull().sum()*100/len(df)
null_percent[lambda x : x>50].round(4).sort_values(ascending=False)

# Cell 7
print("Count of columns with less than 50% fill rate: ",len(null_percent[lambda x : x>50]))

# Cell 9
df.drop(['engineTransmission_engine_cc_value_10', 'engineTransmission_engineOil_cc_value_9'], axis=1, inplace=True, errors='ignore')

# Cell 10
cols = [label for label in df.columns if any(x in label for x in ['comments'])]
df.drop(cols, axis = 1,inplace = True)

# Cell 11
# Impute nulls in all of the columns with 'current condition if not yes' as per data dictionary
cols = [label for label in df.columns if any(x in label for x in ['engineTransmission'])]
df[cols] = df[cols].fillna('yes')

# Cell 12
print("Number of Categorical columns: ", df.select_dtypes('O').shape[1])
print("Number of Continous columns: ", df.select_dtypes(include=['float64','int64']).shape[1])
print("Number of Timestamp columns: ", df.select_dtypes('datetime64').shape[1])

# Cell 14
if 'appointmentId' in df.columns:
    is_unique = df['appointmentId'].nunique() == df['appointmentId'].count()
    print(f"Is appointmentId unique? {is_unique}")
else:
    print("appointmentId column not found (likely already dropped or set as index).")

# Cell 15
# Print all columns to find the right name
print([col for col in df.columns if 'time' in col.lower()])

# Or print all columns
print(df.columns)

# Cell 16
print("We have", (df['inspectionStartTime'].max()-df['inspectionStartTime'].min()).days, "days of data")

# Cell 17
# Extract the inspection month, date, day of week & hour
df['inspection_hour'] = df['inspectionStartTime'].dt.hour
df['inspection_mon'] = df['inspectionStartTime'].dt.month
df['inspection_date'] = df['inspectionStartTime'].dt.date
df['inspection_dow'] = df['inspectionStartTime'].dt.day_name()

# Cell 18
timeSeries = df.groupby(['inspection_date']).count()['appointmentId'].reset_index()
timeSeries = timeSeries.loc[:, ["inspection_date","appointmentId"]]
timeSeries.index = timeSeries.inspection_date
ts = timeSeries.drop("inspection_date",axis=1)

plt.figure(figsize=(10, 5))
plt.plot(df.groupby(['inspection_date'])['appointmentId'].count(), label='Original')
rolmean = ts.rolling(7).mean()
rolstd = ts.rolling(7).std() 
mean = plt.plot(rolmean, color='black', label='Rolling Mean')
std = plt.plot(rolstd, color='green', label = 'Rolling Std')
plt.xlabel("Inspection Date")
plt.ylabel("# Inspections")
plt.title("Number of Inspections over time")
plt.legend()
plt.show()

# Cell 19
plt.figure(figsize=(10, 5))

df.groupby(['inspection_mon'])['appointmentId'].count().plot.bar()
plt.xticks(np.arange(0,4),['Jan','Feb', 'Mar','Apr'])
plt.xlabel("Inspection month")
plt.ylabel("Count of Inspections")

# Cell 20
print("Avg Daily inspections: ", df.groupby(['inspection_date']).count()['appointmentId'].mean())

# Cell 21
dow_ct= df.groupby(['inspection_date','inspection_dow']).count().reset_index()
plt.plot(dow_ct.groupby(['inspection_dow'])['appointmentId'].mean())
plt.xlabel("Inspection Weekday")
plt.ylabel("Average #Inspections")

# Cell 22
sns.histplot(df.groupby('inspection_date').count()['appointmentId'], kde=True)
plt.xlabel("Daily Inspections")
plt.show()

# Cell 23
sns.histplot(data=df, x='inspection_hour', kde=True, bins=10)
plt.show()

# Cell 25
# Distribution of target column 'rating_engineTransmission'
df['rating_engineTransmission'].value_counts(normalize=True)

# Cell 26
sns.histplot(data=df, x='rating_engineTransmission', bins=10, kde=True)
plt.show()

# Cell 27
# Distribution of registeration year month in data

plt.figure(figsize=(10, 5))

df.groupby(['year'])['appointmentId'].count().plot.bar()
plt.xlabel("Registration Year")
plt.ylabel("Count of Inspections")

# Cell 28
sns.boxplot(df["year"])

# Cell 30
# Removing outlier using IQR
def remove_outlier(df, col):

  # 1. Compute Quantiles
  q1 = np.quantile(df[col], .25)
  q3 = np.quantile(df[col], .75)

  IQR = q3 - q1

  # 2. Compute the upper & lower limit
  upper_limit = q3 + 1.5 * IQR
  lower_limit = q1 - 1.5 * IQR

  return df[(df[col] < upper_limit) & (df[col] > lower_limit)]

df = remove_outlier(df, 'year')

# Cell 31
plt.figure(figsize=(10, 5))

# Option 1: Define the color variable
color = 'skyblue' 

# Option 2: Pass the string directly: .plot.bar(color='skyblue')
df.groupby(['month'])['appointmentId'].count().plot.bar(color=color)

plt.xlabel("Registration Month")
plt.ylabel("Count of Inspections")

# Cell 33
sns.boxplot(df["odometer_reading"])

# Cell 35
# Removing outliers using IQR
df = remove_outlier(df, 'odometer_reading')

# Cell 37
# Visualising pearson's correlation
cols = df.select_dtypes(include=['float64','int64']).columns
corr = df[cols].corr()
plt.figure(figsize=(10, 5))
sns.heatmap(corr, annot=True, cbar=False)
plt.show()

# Cell 38
df.groupby(['month'])['rating_engineTransmission'].mean()

# Cell 40
def analyze_temporal_trends(feature, target='rating_engineTransmission'):
    # Check distribution of the feature
    print(f"Analyzing '{feature}' vs '{target}'")
    grouped = df.groupby(feature)[target].mean()
    print(grouped)

    # Visualize the distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=feature, y=target, data=df)
    plt.title(f"Distribution of {target} by {feature}")
    plt.show()

    # Perform statistical test
    unique_values = df[feature].unique()
    if len(unique_values) > 2:
        # Use Kruskal-Wallis test for >2 groups
        groups = [df[df[feature] == val][target] for val in unique_values]
        stat, p = kruskal(*groups)
        test_name = "Kruskal-Wallis"
    else:
        # Use t-test for 2 groups
        group1 = df[df[feature] == unique_values[0]][target]
        group2 = df[df[feature] == unique_values[1]][target]
        stat, p = ttest_ind(group1, group2)
        test_name = "t-test"

    print(f"{test_name} results: Statistic={stat:.4f}, p-value={p:.4f}")
    if p < 0.05:
        print(f"Significant differences found in {target} across {feature}.")
    else:
        print(f"No significant differences found in {target} across {feature}.")

# Analyze temporal trends for each feature
temporal_features = ['inspection_hour', 'inspection_mon', 'inspection_dow','month','year']
for feature in temporal_features:
    analyze_temporal_trends(feature)

# Cell 41
# Calculate Spearman correlation for 'year' and 'rating_engineTransmission'
year_corr, year_p = spearmanr(df['year'], df['rating_engineTransmission'])
print(f"Spearman Correlation (Year vs Rating): {year_corr:.4f}, p-value: {year_p:.4f}")

# Cell 44
cat_cols = [label for label in df.columns if any(x in label for x in ['engineTransmission_'])]

# Cell 45
df_cat = df[cat_cols]
df_cat = df_cat.reset_index().drop('index', axis=1)
df_cat_encoded = pd.get_dummies(df_cat, dtype=bool, drop_first=True)

# Cell 46
# Check for any constant columns
df_cat_encoded.loc[:, (df_cat_encoded == df_cat_encoded.iloc[0]).all()].columns

# Cell 47
# List all unique suffixes based on column names
suffixes = df_cat_encoded.columns.str.extract(r'.*_(.*)$')[0].unique()

for suffix in suffixes:
    matching_columns = [col for col in df_cat_encoded.columns if f'_{suffix}' in col]
    # remove the numeric part (_0, _1, _2, etc.) and the suffix (_Jump Start) to create the base column name
    base_name = re.sub(r'_\d+_' + re.escape(suffix) + '$', '', matching_columns[0])
    # Combine the selected columns (using OR operation)
    df_cat_encoded[f'{base_name}_{suffix}'] = df_cat_encoded[matching_columns].any(axis=1).astype(int)
    
    # Drop the original individual columns after combining
    df_cat_encoded.drop(columns=matching_columns, inplace=True)

print(df_cat_encoded.shape)

# Cell 48

# Add constant term
X = add_constant(df_cat_encoded)

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)

# Cell 49
# combine encoded columns with df
df= df.reset_index()
df_features = pd.merge(df, df_cat_encoded, left_index=True, right_index=True)
df_features = df_features.drop(cat_cols, axis=1)

combined_columns= [label for label in df_features.columns if any(x in label for x in ['engineTransmission_'])]

df_features[combined_columns] = df_features[combined_columns].astype('O')
df_features['rating_engineTransmission'] = df_features['rating_engineTransmission'].astype(float)

# Cell 50
target = 'rating_engineTransmission'
# Ensure all combined columns are cleaned
for column in combined_columns:
    df_features[column] = df_features[column].fillna('Unknown').astype(str)
for column in combined_columns:
    # Group the target variable by the categories in the column
    groups = [df_features[target][df_features[column] == category] for category in df_features[column].unique()]
     # Filter out categories with fewer than 2 data points
    grouped_data = [group for group in groups if len(group) > 1]
    
    if len(grouped_data) < 2:  # If fewer than two valid groups, skip the ANOVA
        print(f"Skipping ANOVA test for {column}: not enough data in one or more groups.")
        continue
    try:
        # Perform ANOVA test
        f_statistic, p_value = stats.f_oneway(*groups)
        
        # Print results
        print(f"ANOVA Test for Column: {column}")
        print(f"F-Statistic: {f_statistic:.4f}, P-Value: {p_value:.4f}")
        
        # Check significance
        if p_value < 0.05:
            print("Result: Significant relationship with the target variable.\n")
        else:
            print("Result: No significant relationship with the target variable.\n")
    except ValueError as e:
        print(f"Error for column {column}: {e}")

# Cell 51

df_clean = df_features.drop(['appointmentId','inspectionStartTime',
                          'inspection_dow','inspection_date', 'index',
                          'engineTransmission_battery_cc_value_yes', # constant variable
                          'engineTransmission_exhaustSmoke_cc_value_Leakage from manifold'], axis = 1)

# Cell 53
# Split into training set & target
train = df_clean.drop('rating_engineTransmission', axis=1).reset_index().drop('index', axis=1)
target = df_clean['rating_engineTransmission'].reset_index().drop('index', axis=1)

X = train
y = target

# Cell 55

# Identify categorical columns (if necessary)
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)  # One-hot encode categorical features

# Split the data into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Generating KFold Split for cross validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Cell 56
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Print the R squared scores
rf_train_score = rf.score(X_train, y_train)
rf_val_score = rf.score(X_val, y_val)
print(rf_train_score, rf_val_score)

# Cell 57
# Decision Tree
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

# Print the R squared scores
dt_train_score = dt.score(X_train, y_train)
dt_val_score = dt.score(X_val, y_val)
print(dt_train_score, dt_val_score)

# Cell 58
# XGboost
xgb = XGBRegressor()
xgb.fit(X_train, y_train)

# Print the R squared scores
xgb_train_score = xgb.score(X_train, y_train)
xgb_val_score = xgb.score(X_val, y_val)
print(xgb_train_score, xgb_val_score)

# Cell 59
# LGBM Regressor
lgbm = LGBMRegressor()
lgbm.fit(X_train, y_train)

# Print the R squared scores
lgbm_train_score = lgbm.score(X_train, y_train)
lgbm_val_score = lgbm.score(X_val, y_val)
print(lgbm_train_score, lgbm_val_score)

# Cell 61
# Tuning the XGBoost model
# Using RandomizedSearchCV for hyperparameter tuning
parameters = {'max_depth': [3, 5, 6, 10, 15, 20],
              'learning_rate': [0.01, 0.1, 0.2, 0.3],
              'subsample': np.arange(0.5, 1.0, 0.1),
              'colsample_bytree': np.arange(0.4, 1.0, 0.1),
              'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
              'n_estimators': [100, 500, 1000],
             'reg_alpha': [0, 0.01, 0.05],
    'reg_lambda': [0, 0.01, 0.05]}

xgb = XGBRegressor()
rscv_xgb = RandomizedSearchCV(xgb, parameters, n_iter=20,scoring='r2', verbose=30, n_jobs=-1, cv=3, random_state=42)
rscv_xgb.fit(X_train, y_train)

# Cell 62
# Store the best estimator
xgb_best_est = rscv_xgb.best_estimator_
# Compute the R Squared score for the best estimator
train_score = xgb_best_est.score(X_train, y_train)
validation_score = xgb_best_est.score(X_val, y_val)

print(train_score, validation_score)

# Cell 63
# Using RandomizedSearchCV for hyperparameter tuning
parameters = { 'num_leaves': [20, 30, 50],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'objective': ['regression'],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [0, 0.01, 0.1]
              }
lgbm = LGBMRegressor(verbose=-1)
rscv_lgbm = RandomizedSearchCV(lgbm, parameters, n_iter=50,scoring='r2', verbose=3, n_jobs=-1, cv=3, random_state=42)
rscv_lgbm.fit(X_train, y_train)

# Cell 64
# Store the best estimator
lgbm_best_est = rscv_lgbm.best_estimator_
# Compute the R Squared score for the best estimator
train_score = rscv_lgbm.score(X_train, y_train)
validation_score = rscv_lgbm.score(X_val, y_val)

print(train_score, validation_score)

# Cell 65
from lightgbm import LGBMRegressor

# Initialize the Regressor instead of Classifier
lgbm_best_est = LGBMRegressor()

X_train_final = pd.concat((X_train, X_val), axis=0)
y_train_final = pd.concat((y_train, y_val), axis=0)

# Force target to be integer type for Classification
lgbm_best_est.fit(X_train_final, y_train_final.astype(int))

# Cell 66
# Compute R2 Score on test set
final_train_score = lgbm_best_est.score(X_train, y_train)
final_test_score = lgbm_best_est.score(X_test, y_test)

print("Final Train Score: {}".format(final_train_score))
print("Final Test Score: {}".format(final_test_score))

# Cell 67
# Storing the model in a pickle file
final_model = lgbm_best_est
with open('final_model_lgbm.pickle', 'wb') as files:
    pickle.dump(final_model, files)

# Cell 68
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor

# 1. Initialize the Regressor (fixes the 'continuous' label error)
lgbm_best_est = LGBMRegressor()

# 2. Prepare the data (Ensure X_train, X_val, y_train, y_val are loaded from earlier cells)
X_train_final = pd.concat((X_train, X_val), axis=0)
y_train_final = pd.concat((y_train, y_val), axis=0)

# 3. Fit the model
lgbm_best_est.fit(X_train_final, y_train_final)

# 4. Plot Feature Importances
feature_importances = lgbm_best_est.feature_importances_
features = X_train_final.columns

feat_imp_df = pd.DataFrame({'Features': features, 'Feature_Importance': feature_importances})
feat_imp_df = feat_imp_df.sort_values(by='Feature_Importance')

feat_imp_df.plot(kind='barh', x='Features', y='Feature_Importance', figsize=(10, 10))
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()

# Cell 69
feat_imp_df.sort_values(by='Feature_Importance', ascending=False).head(10)


# Cell 70
