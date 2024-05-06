#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


df = pd.read_csv('height-weight.csv')
df.head()


#scatter plot: helps you to point the data points with respect to x and y
plt.scatter(df['Weight'], df['Height'])
plt.xlabel('weight')
plt.ylabel('height')








# Multicollinearity is a statistical phenomenon that occurs when
# two or more independent variables in a regression model are highly correlated with each other.
# In other words, multicollinearity indicates a strong linear relationship among the predictor variables

# In[2]:


"""correlation: finding whether the relationship is positive or negative;
as in this example we can see the relation between weight and height is 0.9 which is so close to 1.
we can also use a library named seaborn for visualization"""
df.corr()


# In[10]:


#seaborn visualization: try to show cordination between x and y 
import seaborn as sns
sns.pairplot(df)

"""multicolinarity: this is a scenario where in if you have two three independent features and the correlation
between those independent features are high then we can neglect those features which are highly correlated.
"""


# In[3]:


# creating independent feature: whenever we try to make a independent feature, we should be careful for our independent
# feature to be in 2 dimensional array or data frame and not a series. 
# type(df) => data frame

X = df[["Weight"]] # independent feature always be in data frame or 2d array
type(X)
"""
for 2d array:
np.array(X)
to see is it a 2d array or not:
np.array(X).shape
"""


# In[4]:


# creating dependent feature: it can be a series or 1d array...no diffrence!

Y = df["Height"]


# In[74]:


# train test split: train data will only be used for training and test data will only be used for testing

from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


# Test Size: This parameter determines the proportion of the dataset that will be allocated to the test set. It can be specified as either a float (between 0.0 and 1.0) or an integer (representing the absolute number of test samples). If you provide a float, it represents the fraction of the dataset to include in the test split. For example, if you set test_size=0.25, it means 25% of the data will be used for testing, and the remaining 75% will be used for training. If you provide an integer, it represents the exact number of test samples.
# Random State: This parameter controls the shuffling applied to the data before splitting. It can take one of the following values:
# An integer: If you pass an integer value, it ensures reproducible output across multiple function calls. In other words, using the same random state value will always result in the same train-test split.
# A RandomState instance: You can also pass an instance of the RandomState class to achieve the same reproducibility.
# None: If you don’t specify a random state, the data will be shuffled randomly each time you call the function.
# In your example, test_size=0.25 means that 25% of the data will be used for testing, and random_state=42 ensures reproducibility in the split. The remaining 75% of the data will be used for training.

# In[76]:


X_train.shape
print(X_test)


# In[13]:


#Standardiziation with z-score formula: z-score = (x-M)/mu  where M=0 and mu(standard deviation)=1
# we just standardize our input feature and not the output feature because gradient descent get applied to independent feature.
from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler() # this is our z-score formula
X_train_scaled = scaler.fit_transform(X_train) # fit_transform is only used for training data input
X_test_scaled = scaler.transform(X_test)
# we use transform func for testing dataset to use the M(mean) and mu(standard deviation) we already had from training data
# because we dont want something named data leakage happens. Data leakage is when information from outside the training dataset is used to create the model.


# you need to ensure that the test data (X_test) is properly standardized using the same scaler that was fitted on the training data.
# By fitting the scaler on the training data and then using the same scaler to transform both the training and test data, you ensure that the scaling is consistent and based on the statistics learned from the training data.

# In[94]:


# apply linear regression
from sklearn.linear_model import LinearRegression
#remember: whenever we are using any libraries from sklearn, first we need to initialize that library
regression = LinearRegression(n_jobs=-1)


# In[95]:


print(Y_train)
regression.fit(X_train_scaled,Y_train)


# In[96]:


print("coeficience:",regression.coef_) #if we had y=ax+b for one feature , this coef_ is our a. and b would be intercept.
print("intercept:",regression.intercept_)


# In[97]:


#plot the training data plot best fit line

plt.scatter(X_train_scaled,Y_train)
plt.plot(X_train_scaled, regression.predict(X_train_scaled)) #here we get the best fit line for out training data


# In[98]:


#prediction for test data
Y_pred = regression.predict(X_test_scaled)


# In[99]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[100]:


mse=mean_squared_error(Y_test,Y_pred)
mae=mean_absolute_error(Y_test,Y_pred)
rmse=np.sqrt(mse)
print("mse",mse)
print("mae",mae)
print("rmse",rmse)


# ## R square 
#  Formula
#  
#  It measures the proportion of the variation in your dependent variable explained by all of your independent variables in the model. It assumes that every independent variable in the model helps to explain variation in the dependent variable. In reality, some independent variables (predictors) don't help to explain dependent (target) variable. In other words, some variables do not contribute in predicting target variable.
#  
# 
# **R^2 = 1 - SSR/SST**
# 
#  
# *R^2	=	coefficient of determination
# 
# *SSR	=	sum of squares regression or residuals(residuals for each observation is the difference between predicted values of 
# y(dependent variable) and observed values of y.
# 
# *SST	=	total sum of squares
# 
# 
# 
# *The sum squared regression is the sum of the residuals squared, and the total sum of squares is the sum of the distance the data is away from the mean all squared. As it is a percentage it will take values between 
# 0 and 1.

# In[101]:


from sklearn.metrics import r2_score


# In[102]:


score = r2_score(Y_test,Y_pred)
print(score)


# **Adjusted R2 = 1 – [(1-R2)*(n-1)/(n-k-1)]**
#  
# where:
#  
# R2: The R2 of the model
# n: The number of observations
# k: The number of predictor variables

# In[103]:


1-(1-score)*(len(Y_test)-1)/(len(Y_test)-X_test_scaled.shape[1]-1)


# In[104]:


# ols in linear regression
import statsmodels.api as sm
model = sm.OLS(Y_train,X_train_scaled).fit()


# In[105]:


prediction = model.predict(X_test_scaled)
print(prediction)


# In[33]:


print(model.summary()) # with the help of ols we can also get good values


# In[106]:


#prediction for new data
new_data = np.array([[72]])  # Create a 2D array
scaled_data = scaler.transform(new_data)  # Scale the new data
prediction = regression.predict(scaled_data)  # Predict using the model
print(prediction)


# In[ ]:




