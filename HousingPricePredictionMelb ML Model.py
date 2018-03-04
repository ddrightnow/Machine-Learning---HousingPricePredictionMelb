import pandas as pd

# Load data
melb_data = pd.read_csv('../input/melb_data.csv/melb_data.csv')
melb_data2 = pd.read_csv('../input/melb_data.csv/melb_data.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

melb_target = melb_data.Price
melb_target2 = melb_data.Price


#print(melb_data.columns)
melb_predictors = melb_data.drop(['Price'], axis=1)

melb_predictors2 = melb_data2.drop(['Price'],1)
#print(melb_predictors2.describe)
# For the sake of keeping the example simple, we'll use only numeric predictors. 
melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])
melb_numeric_predictors2 = melb_predictors2.select_dtypes(exclude=['object'])

y = melb_target
X = melb_numeric_predictors
print(7*8)

#Create Function to Measure Quality of An Approach¶
# def score_dataset(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
def score_dataset(predictors_train, predictors_val, targ_train, targ_val):    
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)


X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0) 

#Get Model Score from Dropping Columns with Missing Values
cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))
#score_dataset(50,reduced_X_train, reduced_X_test, y_train, y_test)

# for max_leaf_nodes in [5,40, 50,100,200,500,1000,5000]:
#     my = score_dataset(max_leaf_nodes,reduced_X_train, reduced_X_test, y_train, y_test)
#     print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my))
   
#Get Model Score from Imputation
from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))

#Get Score from Imputation with Extra Columns Showing What Was Imputed

imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))

#print(X_train[1])
print(imputed_X_train[1])
print(imputed_X_train_plus[1])

