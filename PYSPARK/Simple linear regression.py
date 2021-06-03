#!/usr/bin/env python
# coding: utf-8

# # Simple linear regression

# In[1]:


from pyspark.sql import SparkSession


# In[2]:


spark=SparkSession.builder.appName('training').getOrCreate()


# In[8]:


training=spark.read.csv('test.csv',header=True,inferSchema=True)


# In[9]:


training.show()


# In[12]:


training.printSchema()


# In[13]:


training.columns


# In[14]:


from pyspark.ml.feature import VectorAssembler


# In[15]:


output=VectorAssembler(inputCols=['age','Expreince'],outputCol='independent feature')


# In[16]:


output=output.transform(training)


# In[17]:


output.show()


# In[18]:


finalized_data=output.select('independent feature','salary')


# In[19]:


finalized_data.show()


# In[22]:


train_data,test_data=finalized_data.randomSplit([0.75,0.25])


# In[23]:


from pyspark.ml.regression import LinearRegression


# In[24]:


regressor=LinearRegression(featuresCol='independent feature',labelCol='salary')


# In[25]:


regressor=regressor.fit(train_data)


# In[26]:


pred_result=regressor.evaluate(test_data)


# In[30]:


pred_result.predictions.show()


# In[31]:


pred_result.meanAbsoluteError


# In[32]:


pred_result.meanSquaredError


# In[33]:


regressor.coefficients


# In[34]:


regressor.intercept


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




