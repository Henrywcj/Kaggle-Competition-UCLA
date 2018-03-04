# load the libraries we are using 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy 
import sklearn
from sklearn.preprocessing import LabelEncoder

#%matplotlib inline 
# load the data
train_data_frame = pd.read_csv("train.csv", index_col = 0)
test_data_frame = pd.read_csv("test.csv", index_col = 0)

# dealing with outliers
plt.hist(train_data_frame["GrLivArea"])
print (train_data_frame[train_data_frame["GrLivArea"]>4000])
# a few houses has general living area more than 4000, we drop it (TODO: other variables 
train_data_frame.drop(train_data_frame[train_data_frame["GrLivArea"]>4000].index, inplace=True)


# drop columns according to the rules below: 
# 1. drop "LotFrontage": to many NA values (use alternative, preserve information)
# 2. drop "MSSubClass": information already reflected in other variables (e.g. BldgType, HouseStyle, YearBuilt, YearRemodAdd)
# 3. drop "Utilities": testing data all the same - AllPub (No information)
train_data_frame_trimmed = train_data_frame.drop(['MSSubClass', 'Utilities'], axis = 1)
# do the same to test dataset 
test_data_frame_trimmed = test_data_frame.drop(['MSSubClass', 'Utilities'], axis = 1)

# Alternatively, filling LotFrontage with median based on neighborhood
lot_frontage_group_train = train_data_frame['LotFrontage'].groupby(train_data_frame["Neighborhood"])
for key, group in lot_frontage_group_train: 
    flag = (train_data_frame_trimmed['Neighborhood'] == key) & (train_data_frame_trimmed['LotFrontage'].isnull())
    train_data_frame_trimmed.loc[flag, 'LotFrontage'] = group.median()
# do it the testing in the same way
lot_frontage_group_test = test_data_frame['LotFrontage'].groupby(test_data_frame["Neighborhood"])
for key, group in lot_frontage_group_test: 
    flag = (test_data_frame_trimmed['Neighborhood'] == key) & (test_data_frame_trimmed['LotFrontage'].isnull())
    test_data_frame_trimmed.loc[flag, 'LotFrontage'] = group.median()


# Deal with NA in columns
# first deal with numerical columns
train_data_frame_trimmed['MasVnrArea'].fillna(0.0, inplace = True)
train_data_frame_trimmed['GarageYrBlt'].fillna(1980.0, inplace = True)
test_data_frame_trimmed['MasVnrArea'].fillna(0.0, inplace = True)
test_data_frame_trimmed['GarageYrBlt'].fillna(1980.0, inplace = True)

# some of the NAs are created because the house doesn't have the feature, rather than missing data
# for ordinal columns, transform it into numbers
# With Missing values

levels = {'No' : 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ord_list = ['GarageQual','GarageCond','FireplaceQu', 'BsmtCond', 'BsmtQual', 'PoolQC']
for name in ord_list:
    train_data_frame_trimmed[name].fillna('No', inplace = True)
    train_data_frame_trimmed[name] = train_data_frame_trimmed[name].map(levels).astype(int)
    
train_data_frame_trimmed['BsmtExposure'].fillna('N', inplace = True)
train_data_frame_trimmed['BsmtExposure'] = train_data_frame_trimmed['BsmtExposure'].map(
    {'N': 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)

levels_2 = {'N': 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
train_data_frame_trimmed['BsmtFinType1'].fillna('N', inplace = True)
train_data_frame_trimmed['BsmtFinType1'] = train_data_frame_trimmed['BsmtFinType1'].map(levels_2).astype(int)
train_data_frame_trimmed['BsmtFinType2'].fillna('N', inplace = True)
train_data_frame_trimmed['BsmtFinType2'] = train_data_frame_trimmed['BsmtFinType2'].map(levels_2).astype(int)

train_data_frame_trimmed['GarageFinish'].fillna('N', inplace = True)
train_data_frame_trimmed['GarageFinish'] = train_data_frame_trimmed['GarageFinish'].map(
    {'N': 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype(int)


train_data_frame_trimmed['Fence'].fillna('N', inplace = True)
train_data_frame_trimmed['Fence'] = train_data_frame_trimmed['Fence'].map(
    {'N': 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)
# without missing values, but clearly ordinal 
ordinal_list = ['ExterQual','ExterCond','HeatingQC', 'KitchenQual']
for name in ordinal_list: 
    train_data_frame_trimmed[name] = train_data_frame_trimmed[name].map(levels).astype(int)

train_data_frame_trimmed['Functional'] = train_data_frame_trimmed['Functional'].map(
    {None: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, 
         "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)

# Specialized treatment for test dataset
test_data_frame_trimmed['MSZoning'].fillna('RL', inplace = True) # mode
test_data_frame_trimmed['Exterior1st'].fillna('Viny1Sd', inplace = True) # mode
test_data_frame_trimmed['Exterior2nd'].fillna('VinylSd', inplace = True) # mode
test_data_frame_trimmed['BsmtFinSF1'].fillna(439.0, inplace = True) #mean
test_data_frame_trimmed['BsmtFinSF2'].fillna(53.0, inplace = True) #mean
test_data_frame_trimmed['BsmtUnfSF'].fillna(554.0, inplace = True) #mean
test_data_frame_trimmed['TotalBsmtSF'].fillna(1046.0, inplace = True) #mean
test_data_frame_trimmed['BsmtFullBath'].fillna(0.0, inplace = True) #Mode & mean
test_data_frame_trimmed['BsmtHalfBath'].fillna(0.0, inplace = True)# MOde & mean
test_data_frame_trimmed['KitchenQual'].fillna('TA', inplace = True) # mode
test_data_frame_trimmed['Functional'].fillna('Typ', inplace = True)# mode
test_data_frame_trimmed['GarageCars'].fillna(2.0, inplace = True) # mode
test_data_frame_trimmed['GarageArea'].fillna(473.0, inplace = True) # mean
test_data_frame_trimmed['SaleType'].fillna('WD', inplace = True) # mode

# Same for the testing dataset (just copy & paste from above cell)
# some of the NAs are created because the house doesn't have the feature, rather than missing data
# for ordinal columns, transform it into numbers
# With Missing values

levels = {'No' : 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ord_list = ['GarageQual','GarageCond','FireplaceQu', 'BsmtCond', 'BsmtQual', 'PoolQC']
for name in ord_list:
    test_data_frame_trimmed[name].fillna('No', inplace = True)
    test_data_frame_trimmed[name] = test_data_frame_trimmed[name].map(levels).astype(int)
    
test_data_frame_trimmed['BsmtExposure'].fillna('N', inplace = True)
test_data_frame_trimmed['BsmtExposure'] = test_data_frame_trimmed['BsmtExposure'].map(
    {'N': 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)

levels_2 = {'N': 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
test_data_frame_trimmed['BsmtFinType1'].fillna('N', inplace = True)
test_data_frame_trimmed['BsmtFinType1'] = test_data_frame_trimmed['BsmtFinType1'].map(levels_2).astype(int)
test_data_frame_trimmed['BsmtFinType2'].fillna('N', inplace = True)
test_data_frame_trimmed['BsmtFinType2'] = test_data_frame_trimmed['BsmtFinType2'].map(levels_2).astype(int)

test_data_frame_trimmed['GarageFinish'].fillna('N', inplace = True)
test_data_frame_trimmed['GarageFinish'] = test_data_frame_trimmed['GarageFinish'].map(
    {'N': 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype(int)


test_data_frame_trimmed['Fence'].fillna('N', inplace = True)
test_data_frame_trimmed['Fence'] = test_data_frame_trimmed['Fence'].map(
    {'N': 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)
# without missing values, but clearly ordinal 
ordinal_list = ['ExterQual','ExterCond','HeatingQC', 'KitchenQual']
for name in ordinal_list: 
    test_data_frame_trimmed[name] = test_data_frame_trimmed[name].map(levels).astype(int)

test_data_frame_trimmed['Functional'] = test_data_frame_trimmed['Functional'].map(
    {None: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, 
         "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)


# Others
train_data_frame_trimmed['Alley'].fillna('No Alley', inplace = True)
train_data_frame_trimmed['MiscFeature'].fillna('No MiscFeatures', inplace = True)
train_data_frame_trimmed['MasVnrType'].fillna('None', inplace = True)
train_data_frame_trimmed['GarageType'].fillna('Attchd', inplace = True)
train_data_frame_trimmed['Electrical'].fillna('Sbrkr', inplace = True)
# Transform Central Air to 0,1 
train_data_frame_trimmed["CentralAir"] = train_data_frame_trimmed["CentralAir"].astype('category')
train_data_frame_trimmed["CentralAir"] = train_data_frame_trimmed["CentralAir"].cat.codes

# Misunderstanding
test_data_frame_trimmed['Alley'].fillna('No Alley', inplace = True)
test_data_frame_trimmed['MiscFeature'].fillna('No MiscFeatures', inplace = True)
test_data_frame_trimmed['MasVnrType'].fillna('None', inplace = True)
test_data_frame_trimmed['GarageType'].fillna('Attchd', inplace = True)
test_data_frame_trimmed['Electrical'].fillna('Sbrkr', inplace = True)
# Transform Central Air to 0,1 
test_data_frame_trimmed["CentralAir"] = test_data_frame_trimmed["CentralAir"].astype('category')
test_data_frame_trimmed["CentralAir"] = test_data_frame_trimmed["CentralAir"].cat.codes

cat_variables = []
num_variables = []
for name in train_data_frame_trimmed.columns:
    # print (training_features[name].dtype.name)
    if train_data_frame_trimmed[name].dtype.name == 'object':
        cat_variables.append(name)
    else:
        num_variables.append(name)
print (cat_variables)


for name in cat_variables:
    print (name)
    # for convenience, didn't show distribution
    #plt.hist(train_data_frame_trimmed[name])
    #plt.show()


# binarize some columns
train_data_frame_trimmed['Alley'] = train_data_frame_trimmed['Alley'].map(
    {'No Alley':0, 'Grvl':1, 'Pave':1}).astype(int)
test_data_frame_trimmed['Alley'] = test_data_frame_trimmed['Alley'].map(
    {'No Alley':0, 'Grvl':1, 'Pave':1}).astype(int)
train_data_frame_trimmed['LotShape'] = train_data_frame_trimmed['LotShape'].map(
    {'Reg':0, 'IR1':1, 'IR2':1, 'IR3':1}).astype(int)
test_data_frame_trimmed['LotShape'] = test_data_frame_trimmed['LotShape'].map(
    {'Reg':0, 'IR1':1, 'IR2':1, 'IR3':1}).astype(int)
train_data_frame_trimmed['LandSlope'] = train_data_frame_trimmed['LandSlope'].map(
    {'Gtl':0, 'Mod':1, 'Sev':1}).astype(int)
test_data_frame_trimmed['LandSlope'] = test_data_frame_trimmed['LandSlope'].map(
    {'Gtl':0, 'Mod':1, 'Sev':1}).astype(int)
train_data_frame_trimmed['Condition1'] = train_data_frame_trimmed['Condition1'].map(
    {'Norm':0, 'Artery':1, 'Feedr':1, 'PosA':1, 'PosN': 1, 'RRAe':1 ,'RRAn': 1, 'RRNe': 1, 'RRNn': 1}).astype(int)
test_data_frame_trimmed['Condition1'] = test_data_frame_trimmed['Condition1'].map(
    {'Norm':0, 'Artery':1, 'Feedr':1, 'PosA':1, 'PosN': 1, 'RRAe':1 ,'RRAn': 1, 'RRNe': 1, 'RRNn': 1}).astype(int)

train_data_frame_trimmed.to_csv('train_trimmed.csv')
test_data_frame_trimmed.to_csv('test_trimmed.csv')


# MODELING PART
# import libraries and data
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA

training_df = pd.read_csv('train_trimmed.csv',index_col=0)
testing_df = pd.read_csv('test_trimmed.csv', index_col=0)
label = np.log(training_df['SalePrice']) # log price -- convention 
training_features = training_df.drop(['SalePrice'], axis = 1)
# correlation Part
# observe there is quite strong correlation between variables 
# so we apply PCA to handle this 
# However we will apply PCA after one-hot encoding
correlation_matrix = training_df.corr().abs()
s = correlation_matrix.unstack()
so = s.sort_values(kind="quicksort")
so = so[:3192]
print (training_df.shape)
print (so[3150:])
# split the dataset into training and validation
valid_df = training_features.iloc[:100]
training_df = training_features.iloc[100:]
valid_label = label.iloc[:100]
training_label = label.iloc[100:]

# categorical variables --> one hot encoding

# find out which columns is categorical
cat_variables = []
num_variables = []
for name in training_df.columns:
    # print (training_features[name].dtype.name)
    if training_df[name].dtype.name == 'object':
        cat_variables.append(name)
    else:
        num_variables.append(name)
print (cat_variables)

cat_dict = training_df[ cat_variables ].to_dict( orient = 'records' )
cat_dict_valid = valid_df[ cat_variables ].to_dict( orient = 'records' )
cat_dict_test = testing_df[cat_variables].to_dict(orient = 'records')

training_numerical = training_df[ num_variables ].as_matrix()
valid_numerical = valid_df[ num_variables].as_matrix()
testing_numerical = testing_df[num_variables]
#max_train = np.amax( training_numerical, 0 )

#x_num_train = training_numerical / max_train
#x_num_valid = valid_numerical/ max_train
#x_num_test = testing_numerical / max_train     # scale test by max_train
from sklearn.feature_extraction import DictVectorizer as DV
# vectorize

vectorizer = DV( sparse = False )
vec_x_cat_train = vectorizer.fit_transform( cat_dict )
vec_x_cat_test = vectorizer.transform( cat_dict_test)
vec_x_cat_valid = vectorizer.transform (cat_dict_valid)
training_complete = np.hstack((training_numerical,vec_x_cat_train))
valid_complete = np.hstack((valid_numerical,vec_x_cat_valid))
testing_complete = np.hstack((testing_numerical,vec_x_cat_test))

print (training_complete.shape)
print (valid_complete.shape)
print (testing_complete.shape)
# # now we have one-hot encoding and normalized numerical variable 
# # next step we apply BernoulliRBM 
# from sklearn.neural_network import BernoulliRBM
# RBM = BernoulliRBM(n_components=80)
# training_complete_RBM_transform = RBM.fit_transform(training_complete)
# testing_complete_RBM_transform = RBM.fit_transform(test_complete)

# Apply PCA here
pca = PCA(n_components=178) 
training_complete_1 = pca.fit_transform(training_complete)
valid_complete_1 = pca.transform(valid_complete)
testing_complete_1 = pca.transform(testing_complete)
print (training_complete_1.shape)
print (valid_complete_1.shape)
print (testing_complete_1.shape)
print ('Explained variation per principal component (PCA): {}'.format(np.sum(pca.explained_variance_ratio_)))

error = []
# Linear Regression -- used as benchmark
linear_regression = LinearRegression()
linear_regression.fit(training_complete_1, training_label)
pred_labels = linear_regression.predict(testing_complete_1)
pred_valid = linear_regression.predict(valid_complete_1)
pred_train = linear_regression.predict(training_complete_1)
print ("result: Simple Linear Regression")
print (mean_squared_error(pred_train, training_label))
print (mean_squared_error(pred_valid, valid_label))
error.append(mean_squared_error(pred_valid, valid_label))
d = {'id': testing_numerical.index.values, 'SalePrice':np.exp(pred_labels) }
submission = pd.DataFrame(data = d)
submission.set_index(keys = 'id', inplace = True)
submission.to_csv('submission.csv')

# still linear model - LASSO regression
linear_regression_lasso = Lasso(alpha = 0.001, max_iter=50000)

linear_regression_lasso.fit(training_complete_1, training_label)
pred_labels_1 = linear_regression_lasso.predict(testing_complete_1)
pred_valid_1 = linear_regression_lasso.predict(valid_complete_1)
pred_train = linear_regression_lasso.predict(training_complete_1)
print ("result: LASSO")
print (mean_squared_error(pred_train, training_label))
print (mean_squared_error(pred_valid_1, valid_label))
error.append(mean_squared_error(pred_valid, valid_label))

# still linear model - RIDGE regression
linear_regression_ridge = Ridge(alpha = 0.39, max_iter=50000)
linear_regression_ridge.fit(training_complete_1, training_label)
pred_labels_2 = linear_regression_ridge.predict(testing_complete_1)
pred_valid_2 = linear_regression_ridge.predict(valid_complete_1)
pred_train = linear_regression_ridge.predict(training_complete_1)
print ("result: RIDGE")
print (mean_squared_error(pred_train, training_label))
print (mean_squared_error(pred_valid_2, valid_label))
error.append(mean_squared_error(pred_valid, valid_label))

# still linear model - ElasticNet regression
linear_regression_ElasticNet = ElasticNet(alpha = 0.0006, max_iter=50000, l1_ratio = 0.53)

linear_regression_ElasticNet.fit(training_complete_1, training_label)
pred_labels_3 = linear_regression_ElasticNet.predict(testing_complete_1)
pred_valid_3 = linear_regression_ElasticNet.predict(valid_complete_1)
pred_train = linear_regression_ElasticNet.predict(training_complete_1)
print ("result: ELASTICNET")
print (mean_squared_error(pred_train, training_label))
print (mean_squared_error(pred_valid_3, valid_label))
error.append(mean_squared_error(pred_valid, valid_label))

# Gradient Boosting 
# Cross validation (n_estimators)

print ("running cross validation on Gradient Boosting - gonna take a while")

n_est = [130, 140, 150]
max_dep = [6, 7, 8, 9]

er = 1111
for n_e in n_est:
    for m_d in max_dep:
        regre = GradientBoostingRegressor(learning_rate=0.05,
                                 max_depth=6, n_estimators = 100)
        regre.fit(training_complete, training_label)
        pred_labels = regre.predict(testing_complete)
        pred_valid = regre.predict(valid_complete)
        pred_train = regre.predict(training_complete)
        print (mean_squared_error(pred_train, training_label))
        print (mean_squared_error(pred_valid, valid_label))
        if mean_squared_error(pred_valid, valid_label) < er: 
            best_n_e = n_e
            best_m_d = m_d
            result_XGB = pred_labels
            results_valid = pred_valid
            er = mean_squared_error(pred_valid, valid_label)
print (best_n_e, best_m_d)

# adding up three results
results = (pred_labels_1 + pred_labels_2 + pred_labels_3)/3
valid_results = (pred_valid_1 + pred_valid_2 + pred_valid_3)/3
print (mean_squared_error(valid_results, valid_label))
# d = {'id': testing_numerical.index.values, 'SalePrice':np.exp(results) }
# submission = pd.DataFrame(data = d)
# submission.set_index(keys = 'id', inplace = True)
# submission.to_csv('submission.csv')

valid_results_1 = (3* valid_results + results_valid)/4
print (mean_squared_error(valid_results_1, valid_label))
results_1 = (3 * results + result_XGB)/4
d = {'id': testing_numerical.index.values, 'SalePrice':np.exp(results_1) }
submission = pd.DataFrame(data = d)
submission.set_index(keys = 'id', inplace = True)
submission.to_csv('submission.csv')