#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering and Modelling
# 
# ---
# 
# 1. Import packages
# 2. Load data
# 3. Modelling
# 
# ---
# 
# ## 1. Import packages

# In[5]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

# Shows plots in jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Set plot style
sns.set(color_codes=True)


# ---
# ## 2. Load data

# In[7]:


df = pd.read_csv(r"C:\Users\parid\Downloads\data_for_predictions.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)
df.head()


# ---
# 
# ## 3. Modelling
# 
# We now have a dataset containing features that we have engineered and we are ready to start training a predictive model. Remember, we only need to focus on training a `Random Forest` classifier.

# In[8]:


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ### Data sampling
# 
# The first thing we want to do is split our dataset into training and test samples. The reason why we do this, is so that we can simulate a real life situation by generating predictions for our test sample, without showing the predictive model these data points. This gives us the ability to see how well our model is able to generalise to new data, which is critical.
# 
# A typical % to dedicate to testing is between 20-30, for this example we will use a 75-25% split between train and test respectively.

# In[9]:


# Make a copy of our data
train_df = df.copy()

# Separate target variable from independent variables
y = df['churn']
X = df.drop(columns=['id', 'churn'])
print(X.shape)
print(y.shape)


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ### Model training
# 
# Once again, we are using a `Random Forest` classifier in this example. A Random Forest sits within the category of `ensemble` algorithms because internally the `Forest` refers to a collection of `Decision Trees` which are tree-based learning algorithms. As the data scientist, you can control how large the forest is (that is, how many decision trees you want to include).
# 
# The reason why an `ensemble` algorithm is powerful is because of the laws of averaging, weak learners and the central limit theorem. If we take a single decision tree and give it a sample of data and some parameters, it will learn patterns from the data. It may be overfit or it may be underfit, but that is now our only hope, that single algorithm. 
# 
# With `ensemble` methods, instead of banking on 1 single trained model, we can train 1000's of decision trees, all using different splits of the data and learning different patterns. It would be like asking 1000 people to all learn how to code. You would end up with 1000 people with different answers, methods and styles! The weak learner notion applies here too, it has been found that if you train your learners not to overfit, but to learn weak patterns within the data and you have a lot of these weak learners, together they come together to form a highly predictive pool of knowledge! This is a real life application of many brains are better than 1.
# 
# Now instead of relying on 1 single decision tree for prediction, the random forest puts it to the overall views of the entire collection of decision trees. Some ensemble algorithms using a voting approach to decide which prediction is best, others using averaging. 
# 
# As we increase the number of learners, the idea is that the random forest's performance should converge to its best possible solution.
# 
# Some additional advantages of the random forest classifier include:
# 
# - The random forest uses a rule-based approach instead of a distance calculation and so features do not need to be scaled
# - It is able to handle non-linear parameters better than linear based models
# 
# On the flip side, some disadvantages of the random forest classifier include:
# 
# - The computational power needed to train a random forest on a large dataset is high, since we need to build a whole ensemble of estimators.
# - Training time can be longer due to the increased complexity and size of thee ensemble

# In[12]:


# Add model training in here!
model = RandomForestClassifier(n_estimators = 100, random_state = 42) # Add parameters to the model!
model.fit(X_train, y_train) # Complete this method call!


# ### Evaluation
# 
# Now let's evaluate how well this trained model is able to predict the values of the test dataset.

# In[19]:


# Generate predictions here!
y_pred = model.predict(X_test)
print((y_test,y_pred))


# In[22]:


# Calculate performance metrics here!
#accuracy refers to correctly measured output from the entire value.  
#Precision is the ratio of correctly predicted positive observations to the total predicted positives.
#Recall The ratio of correctly predicted positive observations to all observations in the actual class.
#F1-Score: The weighted average of Precision and Recall. It considers both false positives and false negatives.
#The number of actual occurrences of each class in the dataset.
from sklearn.metrics import accuracy_score               
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[23]:


# Our model gives an accuracy of 90 percent and therefore, it is an apt model to use.I do think that our model is satisfactory

# Get feature importances
feature_importances = model.feature_importances_

# Create a DataFrame for better visualization
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)


# In[24]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))
plt.pie(importance_df['Importance'], labels=importance_df['Feature'], autopct='%1.1f%%', startangle=140)
plt.title('Feature Importance Distribution')
plt.show()


# In[28]:


plt.savefig('feature_importance_pie_chart.png', format='png', bbox_inches='tight')


# In[29]:


import os
print(f"Current working directory: {os.getcwd()}")
plt.close()


# In[ ]:




