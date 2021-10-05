


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import warnings
from scipy.stats import skew
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import norm
from collections import Counter
from sklearn import tree
from sklearn.linear_model import LinearRegression, LassoCV, Ridge, LassoLarsCV,ElasticNetCV
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler, LabelEncoder
warnings.filterwarnings('ignore')
sns.set(style='white', context='notebook', palette='deep')
get_ipython().magic("config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook")
get_ipython().magic('matplotlib inline')


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error


train = pd.read_csv("~/Dropbox/2020 Spring/8650data mining/project/train.csv")
test = pd.read_csv("~/Dropbox/2020 Spring/8650data mining/project/test.csv")



print("The train data size before dropping Id feature is : {} ".format(train.shape))

# Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

# Now drop the 'Id' column since it's unnecessary for the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

# Check data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))


# In[5]:


train.head()


# In[6]:


train.info


# In[7]:


train['SalePrice'].describe()


# In[8]:


sns.distplot(train['SalePrice'], hist = False, fit = None);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())


# **explore correlation between numerical variables and saleprices**

# In[9]:


# Correlation Matrix Heatmap
corrmat = train.corr()
f, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(corrmat, vmax=.8, square=True,cmap="YlGnBu");


# In[10]:


corr=train.corr()["SalePrice"]
corr[np.argsort(corr, axis=0)[::-1]]


# In[11]:


#only numeric data
num_feat=train.columns[train.dtypes!=object]
num_feat=num_feat[1:-1] 
labels = []
values = []
for col in num_feat:
    labels.append(col)
    values.append(np.corrcoef(train[col].values, train.SalePrice.values)[0,1])
    
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(8,10))
rects = ax.barh(ind, np.array(values), color='tomato')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation Coefficients w.r.t Sale Price");


# In[12]:


k = 11 #number of variables for heatmap, 10 variables correlation with saleprice higher than 0.5, so 11
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, cmap="BuPu", fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


correlations=train.corr()
attrs = correlations.iloc[:-1,:-1] # all except target

threshold = 0.5
important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0])     .unstack().dropna().to_dict()

unique_important_corrs = pd.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
    for key in important_corrs])), 
        columns=['Attribute Pair', 'Correlation'])

    # sorted by absolute value
unique_important_corrs = unique_important_corrs.ix[
    abs(unique_important_corrs['Correlation']).argsort()[::-1]]

unique_important_corrs


# **look at how each key variable relates with saleprice**

# **overall quality**

# In[14]:


# Overall Quality vs Sale Price
var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# In[15]:


fig, ax = plt.subplots()
# plot histogram
ax.hist(train[var])
# set title and labels
ax.set_title('Overall Quality Distribution')
ax.set_xlabel('OverallQuality')
ax.set_ylabel('Frequency')


# In[16]:


plt.barh(train["OverallQual"],width=train["SalePrice"],color="darksalmon")
plt.title("Sale Price vs Overall Quality of house")
plt.ylabel("Overall Quality of house")
plt.xlabel("Sale Price");


# **living area**

# In[17]:


# Living Area vs Sale Price
plt.scatter(x=train['GrLivArea'], y=train['SalePrice'],
            alpha=0.4, edgecolors='w')
#sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'], kind='reg')


# In[18]:


# Removing outliers manually (Two points in the bottom right)
train = train.drop(train[(train['GrLivArea']>4000) 
                         & (train['SalePrice']<300000)].index).reset_index(drop=True)


# In[19]:


# Living Area vs Sale Price
plt.scatter(x=train['GrLivArea'], y=train['SalePrice'],
            alpha=0.4, edgecolors='w')
#sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'], kind='reg')


# In[20]:


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs = train['GrLivArea']
ys = train["OverallQual"]
zs = train['SalePrice']
ax.scatter(xs, ys, zs, s=50, alpha=0.4, edgecolors='w')

ax.set_xlabel('GrLivArea')
ax.set_ylabel('OverallQual')
ax.set_zlabel('SalePrice')


# **garage number**

# In[21]:


# Garage number vs Sale Price
sns.boxplot(x=train['GarageCars'], y=train['SalePrice'])


# In[22]:


# Removing outliers manually (More than 4-cars, less than $300k)
train = train.drop(train[(train['GarageCars']>3) 
                         & (train['SalePrice']<300000)].index).reset_index(drop=True)


# In[23]:


# Garage number vs Sale Price
sns.boxplot(x=train['GarageCars'], y=train['SalePrice'])


# **garage area**

# In[24]:


# Garage Area vs Sale Price
plt.scatter(x=train['GarageArea'], y=train['SalePrice'],
            alpha=0.4, edgecolors='w')
#sns.jointplot(x=train['GarageArea'], y=train['SalePrice'], kind='reg')


# In[26]:


# Removing outliers manually (More than 1000 sqft, less than $300k)
train = train.drop(train[(train['GarageArea']>1000) 
                         & (train['SalePrice']<300000)].index).reset_index(drop=True)


# In[27]:


# Garage Area vs Sale Price
plt.scatter(x=train['GarageArea'], y=train['SalePrice'],
            alpha=0.4, edgecolors='w')
#sns.jointplot(x=train['GarageArea'], y=train['SalePrice'], kind='reg')


# **basement area**

# In[28]:


# Basement Area vs Sale Price
plt.scatter(x=train['TotalBsmtSF'], y=train['SalePrice'],
            alpha=0.4, edgecolors='w')
#sns.jointplot(x=train['TotalBsmtSF'], y=train['SalePrice'], kind='reg')


# In[29]:


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs = train['OverallQual']
ys = train["TotalBsmtSF"]
zs = train['SalePrice']
ax.scatter(xs, ys, zs, s=50, alpha=0.4, edgecolors='w')

ax.set_xlabel('OverallQual')
ax.set_ylabel('TotalBsmtSF')
ax.set_zlabel('SalePrice')


# **first floor area**

# In[30]:


# First Floor Area vs Sale Price
plt.scatter(x=train['1stFlrSF'], y=train['SalePrice'],
            alpha=0.4, edgecolors='w')


# **total rooms**

# In[31]:


# Total Rooms vs Sale Price
sns.boxplot(x=train['TotRmsAbvGrd'], y=train['SalePrice'])


# In[32]:


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs = train['1stFlrSF']
ys = train["TotRmsAbvGrd"]
zs = train['SalePrice']
ax.scatter(xs, ys, zs, s=50, alpha=0.4, edgecolors='w')

ax.set_xlabel('1stFlrSF')
ax.set_ylabel('TotRmsAbvGrd')
ax.set_zlabel('SalePrice')


# In[33]:


#second floor area
plt.scatter(train["2ndFlrSF"],train["SalePrice"], alpha=0.4, edgecolors='w')
plt.title("Sale Price vs 2nd floor in sq feet");
plt.xlabel("2nd floor in sq feet")
plt.ylabel("Sale Price");


# **year built**

# In[34]:


# Year built vs Sale Price
var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


# In[35]:


var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y="SalePrice", ylim=(0, 800000))


# In[36]:


var = 'YearBuilt'
data = pd.concat([train['OverallQual'], train[var]], axis=1)
data.plot.scatter(x=var, y="OverallQual", ylim=(0, 10))


# In[37]:


fig, ax = plt.subplots()
# plot histogram
ax.hist(train[var])
# set title and labels
ax.set_title('YearBuilt Distribution')
ax.set_ylabel('Frequency')
ax.set_xlabel('YearBuilt')


# **FullBath**       

# In[38]:


# Total bathrooms vs Sale Price
sns.boxplot(x=train['FullBath'], y=train['SalePrice'])


# **remodel year**

# In[39]:


# remodel year vs Sale Price
var = 'YearRemodAdd'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


# **nonnumerical data correlation with saleprice**

# In[40]:


#neighborhood
f = plt.subplots(figsize=(25, 8))
fig = sns.boxplot(x=train['Neighborhood'], y=train['SalePrice'])


# In[41]:


#lotshape
f = plt.subplots(figsize=(5, 3))
fig = sns.boxplot(x=train['LotShape'], y=train['SalePrice'])


# In[42]:


#housestyle
f = plt.subplots(figsize=(8, 3))
fig = sns.boxplot(x=train['HouseStyle'], y=train['SalePrice'])


# In[43]:


#foundation
f = plt.subplots(figsize=(8, 3))
fig = sns.boxplot(x=train['Foundation'], y=train['SalePrice'])


# In[44]:


#central air
f = plt.subplots(figsize=(8, 3))
fig = sns.boxplot(x=train['CentralAir'], y=train['SalePrice'])


# In[45]:


#garage type
f = plt.subplots(figsize=(8, 3))
fig = sns.boxplot(x=train['GarageType'], y=train['SalePrice'])


# In[46]:


#most streets are paved lets visulalize it
sns.stripplot(x=train["Street"], y=train["SalePrice"],jitter=True)
plt.title("Sale Price vs Streets");


# In[47]:


#drop high inter-correlated variables (keep one), selected feature
sns.set()
cols = ['SalePrice','OverallQual','GrLivArea', 'GarageCars', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show()


# # deal with missing value

# In[48]:


# Combining Datasets
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("Train data size is : {}".format(train.shape))
print("Test data size is : {}".format(test.shape))
print("Combined dataset size is : {}".format(all_data.shape))


# In[49]:


# Find Missing Ratio of Dataset
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data


# In[50]:


# Percent missing data by feature
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na, palette="Paired")
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# In[51]:


#impute data with data description, frequent value
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")


# In[52]:


#check whether there is missing value
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# # partition based prediction: decision tree and random forest prediction with selected features

# **Decision tree prediction before data normalization**

# In[53]:


# We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
y_train = train["SalePrice"] 


#decision tree prediction
cols = ['OverallQual','GrLivArea', 'GarageCars', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
x = train[cols].values
#y = train['SalePrice']
y = y_train
X_train,X_test, y_train1, y_test1 = train_test_split(x, y, test_size=0.33, random_state=42)

clf = tree.DecisionTreeRegressor()
clf.fit(X_train, y_train1)
y_pred = clf.predict(X_test)
y_pred_over = clf.predict(X_train)


# In[55]:


#mean absolute error
sum(abs(y_pred - y_test1))/len(y_pred)


# In[56]:


sum(abs(y_pred_over - y_train1))/len(y_pred_over)


# In[57]:


#mean absolute percentage error, to show that decision tree model is overfitted, so the data is 
#too complex for a single decision tree
sum(abs(y_pred - y_test1)/y_test1)/len(y_pred)


# In[58]:


sum(abs(y_pred_over - y_train1)/y_train1)/len(y_pred_over)


# In[60]:


#random forest prediction
clf = RandomForestRegressor(n_estimators=400)
clf.fit(X_train, y_train1)
y_pred = clf.predict(X_test)
y_pred_over = clf.predict(X_train)


# In[61]:


#mean absolute error
sum(abs(y_pred - y_test1))/len(y_pred)


# In[62]:


sum(abs(y_pred_over - y_train1))/len(y_pred_over)


# In[63]:


#mean absolute percentage error
sum(abs(y_pred - y_test1)/y_test1)/len(y_pred)


# In[64]:


sum(abs(y_pred_over - y_train1)/y_train1)/len(y_pred_over)


# **apply labelencoder to categorial data**

# In[65]:


cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# Process columns and apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# Check shape        
print('Shape all_data: {}'.format(all_data.shape))
print(all_data['FireplaceQu'])


# **add a useful feature**

# In[66]:


# Adding Total Square Feet feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# In[67]:


from sklearn import preprocessing


# In[68]:


all_data_e = all_data.select_dtypes(exclude=['object'])
all_data_norm = preprocessing.normalize(all_data_e)


# **decision tree and random forest with all numerical and categorical data**

# In[69]:


#decision tree
train2 = all_data_e[:ntrain]

x2 = train2.values
y2 = y_train
X_train2,X_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.33, random_state=42)

clf = tree.DecisionTreeRegressor()
clf.fit(X_train2, y_train2)
y_pred2 = clf.predict(X_test2)
y_pred_over2 = clf.predict(X_train2)


# In[70]:


#mean absolute error
sum(abs(y_pred2 - y_test2))/len(y_pred2)


# In[71]:


sum(abs(y_pred_over2 - y_train2))/len(y_pred_over2)


# In[72]:


#mean absolute percentage error
sum(abs(y_pred2 - y_test2)/y_test2)/len(y_pred2)


# In[73]:


sum(abs(y_pred_over2 - y_train2)/y_train2)/len(y_pred_over2)


# In[74]:


train2 = all_data_e[:ntrain]

x2 = train2.values
y2 = y_train
X_train2,X_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.33, random_state=42)

clf = RandomForestRegressor(n_estimators=400)
clf.fit(X_train2, y_train2)
y_pred2 = clf.predict(X_test2)
y_pred_over2 = clf.predict(X_train2)


# In[75]:


#mean absolute error
sum(abs(y_pred2 - y_test2))/len(y_pred2)


# In[76]:


sum(abs(y_pred_over2 - y_train2))/len(y_pred_over2)


# In[77]:


#mean absolute percentage error
sum(abs(y_pred2 - y_test2)/y_test2)/len(y_pred2)


# In[78]:


sum(abs(y_pred_over2 - y_train2)/y_train2)/len(y_pred_over2)


# In[79]:


#to show normalization does not work well for random forest
train3 = all_data_norm[:ntrain]

x3 = train3
y3 = train['SalePrice']
X_train3,X_test3, y_train3, y_test3 = train_test_split(x3, y3, test_size=0.33, random_state=42)

clf = RandomForestRegressor(n_estimators=400)
clf.fit(X_train3, y_train3)
y_pred3 = clf.predict(X_test3)


# In[80]:


#mean absolute error
sum(abs(y_pred3 - y_test3))/len(y_pred3)


# In[81]:


#mean absolute percentage error
sum(abs(y_pred3 - y_test3)/y_test3)/len(y_pred3)



# In[86]:


train = all_data_e[:ntrain]
test = all_data_e[ntrain:]


# In[87]:


# Cross-validation with k-folds
n_folds = 5

def mae_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)
    #rmse= np.sqrt(-cross_val_score(model, train, y_train, scoring="mean_absolute_error", cv = kf))
    mae= -cross_val_score(model, train, y_train, scoring="neg_mean_absolute_error", cv = kf)
    return(mae)


# In[88]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=3))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.0005, kernel='polynomial', degree=1, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.005,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
# dt = tree.DecisionTreeRegressor()
rf = RandomForestRegressor(n_estimators=400)


# In[89]:


score = mae_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = mae_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = mae_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = mae_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(dt)
# print("DecisionTree: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = mae_cv(rf)
print("RandomForest: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[90]:


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   


# In[95]:


from mlxtend.regressor import StackingCVRegressor


# In[96]:


stack = StackingCVRegressor(regressors=(ENet, lasso, KRR, rf),
                            meta_regressor=GBoost, cv=5,
                            use_features_in_secondary=True,
                            store_train_meta_features=True,
                            shuffle=False,
                            random_state=42)


# In[97]:


score = mae_cv(stack)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# In[ ]:


plt.plot(y_train)


# In[ ]:


from sklearn.model_selection import cross_val_predict
y = y_train
predicted = cross_val_predict(stack, train, y, cv=5)



# In[ ]:


fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Actual Price')
ax.set_ylabel('Predicted Price')
plt.show()


# In[ ]:


print(train)


# In[ ]:


print(train.shape)


# In[ ]:


print(train.loc[1,:])


# In[ ]:


stack.fit(train,y)


# In[ ]:


x_new = train.loc[500,:].values.reshape(1,-1)
y_predict = stack.predict(x_new)


# In[ ]:


print(y_predict)


# In[ ]:


print(y_train[500])


# In[ ]:


index = [1,100,500,1000]
for x in index:
    x_new = train.loc[x,:].values.reshape(1,-1)
    y_predict = stack.predict(x_new)
    print("predict:", y_predict)
    print("actual", y_train[x])

