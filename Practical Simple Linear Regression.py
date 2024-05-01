# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df=pd.read_csv('height-weight.csv')


# In[5]:


df.head()


# In[7]:


##scatter plot
plt.scatter(df['Weight'],df['Height'])
plt.xlabel("Weight")
plt.ylabel("Height")


# In[8]:


## Correlation
df.corr()


# In[9]:


## Seaborn for visualization
import seaborn as sns
sns.pairplot(df)


# In[23]:


## Independent and dependent features
X=df[['Weight']] ### independent features should be data frame or 2 dimesnionalarray
y=df['Height'] ## this variable can be in series or 1d array


# In[22]:


X_series=df['Weight']
np.array(X_series).shape


# In[25]:


np.array(y).shape


# In[26]:


## Train Test Split
from sklearn.model_selection import train_test_split


# In[27]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[30]:


## Standardization
from sklearn.preprocessing import StandardScaler


# In[32]:


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)


# In[33]:


X_test=scaler.transform(X_test)


# In[34]:


X_test


# In[35]:


## Apply Simple Linear Regression
from sklearn.linear_model import LinearRegression


# In[39]:


regression=LinearRegression(n_jobs=-1)


# In[40]:


regression.fit(X_train,y_train)


# In[44]:


print("Coefficient or slope:",regression.coef_)
print("Intercept:",regression.intercept_)


# In[46]:


## plot Training data plot best fit line
plt.scatter(X_train,y_train)
plt.plot(X_train,regression.predict(X_train))


# ### prediction of test data
# 1. predicted height output= intercept +coef_(Weights)
# 2. y_pred_test =156.470 + 17.29(X_test)

# In[49]:


## Prediction for test data
y_pred=regression.predict(X_test)


# In[50]:


## Performance Metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[51]:


mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(mse)
print(mae)
print(rmse)


# ## R square 
# Formula
# 
# **R^2 = 1 - SSR/SST**
# 
# 
# R^2	=	coefficient of determination
# SSR	=	sum of squares of residuals
# SST	=	total sum of squares

# In[52]:


from sklearn.metrics import r2_score


# In[53]:


score=r2_score(y_test,y_pred)
print(score)


# **Adjusted R2 = 1 – [(1-R2)*(n-1)/(n-k-1)]**
# 
# where:
# 
# R2: The R2 of the model
# n: The number of observations
# k: The number of predictor variables

# In[54]:


#display adjusted R-squared
1 - (1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)


# In[55]:


## OLS Linear Regression
import statsmodels.api as sm


# In[56]:


model=sm.OLS(y_train,X_train).fit()


# In[57]:


prediction=model.predict(X_test)
print(prediction)


# In[58]:


print(model.summary())


# In[60]:


## Prediction For new data
regression.predict(scaler.transform([[72]]))


# In[ ]:




