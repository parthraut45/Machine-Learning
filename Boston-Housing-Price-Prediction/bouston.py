#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


    # In[5]:


from sklearn.datasets import load_boston


    # In[2]:



    boston = load_boston()


    # In[7]:


    boston.keys()


    # In[8]:


    print(boston.DESCR)


    # In[9]:


    dataset = pd.DataFrame(boston.data , columns = boston.feature_names)


    # In[10]:


    dataset.head()


    # In[11]:


    dataset['Price'] = boston.target


    # In[12]:


    dataset.head()


    # In[13]:


    #data cleaning
    dataset.info()


    # In[14]:


    dataset.describe() #gives stats


    # In[15]:


    dataset.isnull()


    # In[16]:


    dataset.isnull().sum()


    # In[17]:


    #data analytics using correlation
    dataset.corr()


    # In[18]:


    import seaborn as sns


    # In[19]:


    sns.pairplot(dataset)


    # In[20]:


    plt.scatter(dataset["RM"],dataset["Price"])
    plt.xlabel("RM")
    plt.ylabel("Price")


    # In[21]:


    sns.regplot(x = "RM" , y = "Price", data = dataset)


    # In[22]:


    #independent and dependent features
    X = dataset.iloc[:,:-1]
    Y = dataset.iloc[:,-1]


    # In[23]:


    Y


    # In[24]:


    X


    # In[25]:


    from sklearn.model_selection import train_test_split


    # In[26]:


    X_train,X_test,Y_train,Y_test  = train_test_split(X,Y,test_size = 0.2, random_state = 42)


    # In[27]:


    Y_test


    # In[28]:


    #standardize the datset
    from sklearn.preprocessing import StandardScaler


    # In[29]:


    scaler = StandardScaler()


    # In[30]:


    X_train= scaler.fit_transform(X_train)


    # In[31]:


    X_test = scaler.transform(X_test)


    # In[32]:


    from sklearn.linear_model import LinearRegression


    # In[33]:


    lreg = LinearRegression()


    # In[34]:


    lreg.fit(X_train,Y_train)


    # In[35]:


    print(lreg.coef_)


    # In[36]:


    print(lreg.intercept_)


    # In[37]:


    lreg.get_params()


    # In[10]:


    #prediction
    lreg.predict(X_test)


    # In[40]:





    # In[41]:


    #prediction 
    plt.scatter(Y_test,lreg_pred)


    # In[43]:


    sns.regplot(x = Y_test, y = lreg_pred)


    # In[44]:


    residuals = Y_test - lreg_pred


    # In[45]:


    residuals


    # In[47]:


    #ploting the residuals
    sns.displot(residuals,kind = "kde")


    # In[48]:


    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error


    # In[49]:


    print(mean_squared_error(Y_test,lreg_pred))


    # In[50]:


    print(mean_absolute_error(Y_test,lreg_pred))


    # In[51]:


    print(np.sqrt(mean_squared_error(Y_test,lreg_pred)))


    # In[52]:


    from sklearn.metrics import r2_score


    # In[53]:


    score = r2_score(Y_test, lreg_pred)


    # In[54]:


    print(score)


    # In[57]:


    #New Data Prediction
    boston.data[0].reshape(1,-1)


    # In[58]:


    #standardization of new data
    scaler.transform(boston.data[0].reshape(1,-1))


    # In[59]:


    lreg.predict(boston.data[0].reshape(1,-1))


    # In[60]:


    #Pickling for deployment
    import pickle


    # In[63]:


    pickle.dump(lreg,open("lreg_model.pkl","wb"))


    # In[64]:


    pickled_model = pickle.load(open("lreg_model.pkl","rb"))


    # In[66]:


    pickled_model.predict(scaler.transform(boston.data[0].reshape(1,-1)))


    # In[ ]:




