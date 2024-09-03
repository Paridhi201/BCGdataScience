#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis Starter
# 
# ## Import packages

# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Shows plots in jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Set plot style
sns.set(color_codes=True)


# ---
# 
# ## Loading data with Pandas
# 
# We need to load `client_data.csv` and `price_data.csv` into individual dataframes so that we can work with them in Python. For this notebook and all further notebooks, it will be assumed that the CSV files will the placed in the same file location as the notebook. If they are not, please adjust the directory within the `read_csv` method accordingly.

# In[11]:


client_df = pd.read_csv(r"C:\Users\parid\Downloads\client_data (1).csv")
price_df = pd.read_csv(r"C:\Users\parid\Downloads\price_data (1).csv")


# You can view the first 3 rows of a dataframe using the `head` method. Similarly, if you wanted to see the last 3, you can use `tail(3)`

# In[3]:


client_df.head(3)


# In[4]:


price_df.head(3)


# ---
# 
# ## Descriptive statistics of data
# 
# ### Data types
# 
# It is useful to first understand the data that you're dealing with along with the data types of each column. The data types may dictate how you transform and engineer features.
# 
# To get an overview of the data types within a data frame, use the `info()` method.

# In[5]:


client_df.info()


# In[6]:


price_df.info()


# ### Statistics
# 
# Now let's look at some statistics about the datasets. We can do this by using the `describe()` method.

# In[7]:


client_df.describe()


# In[8]:


price_df.describe()


# ---
# 
# ## Data visualization
# 
# If you're working in Python, two of the most popular packages for visualization are `matplotlib` and `seaborn`. We highly recommend you use these, or at least be familiar with them because they are ubiquitous!
# 
# Below are some functions that you can use to get started with visualizations. 

# In[9]:


def plot_stacked_bars(dataframe, title_, size_=(18, 10), rot_=0, legend_="upper right"):
    """
    Plot stacked bars with annotations
    """
    ax = dataframe.plot(
        kind="bar",
        stacked=True,
        figsize=size_,
        rot=rot_,
        title=title_
    )

    # Annotate bars
    annotate_stacked_bars(ax, textsize=14)
    # Rename legend
    plt.legend(["Retention", "Churn"], loc=legend_)
    # Labels
    plt.ylabel("Company base (%)")
    plt.show()

def annotate_stacked_bars(ax, pad=0.99, colour="white", textsize=13):
    """
    Add value annotations to the bars
    """

    # Iterate over the plotted rectanges/bars
    for p in ax.patches:
        
        # Calculate annotation
        value = str(round(p.get_height(),1))
        # If value is 0 do not annotate
        if value == '0.0':
            continue
        ax.annotate(
            value,
            ((p.get_x()+ p.get_width()/2)*pad-0.05, (p.get_y()+p.get_height()/2)*pad),
            color=colour,
            size=textsize
        )

def plot_distribution(dataframe, column, ax, bins_=50):
    """
    Plot variable distirbution in a stacked histogram of churned or retained company
    """
    # Create a temporal dataframe with the data to be plot
    temp = pd.DataFrame({"Retention": dataframe[dataframe["churn"]==0][column],
    "Churn":dataframe[dataframe["churn"]==1][column]})
    # Plot the histogram
    temp[["Retention","Churn"]].plot(kind='hist', bins=bins_, ax=ax, stacked=True)
    # X-axis label
    ax.set_xlabel(column)
    # Change the x-axis to plain style
    ax.ticklabel_format(style='plain', axis='x')


# Thhe first function `plot_stacked_bars` is used to plot a stacked bar chart. An example of how you could use this is shown below:

# In[10]:


churn = client_df[['id', 'churn']]
churn.columns = ['Companies', 'churn']
churn_total = churn.groupby(churn['churn']).count()
churn_percentage = churn_total / churn_total.sum() * 100
plot_stacked_bars(churn_percentage.transpose(), "Churning status", (5, 5), legend_="lower right")


# The second function `annotate_bars` is used by the first function, but the third function `plot_distribution` helps you to plot the distribution of a numeric column. An example of how it can be used is given below:

# In[11]:


consumption = client_df[['id', 'cons_12m', 'cons_gas_12m', 'cons_last_month', 'imp_cons', 'has_gas', 'churn']]

fig, axs = plt.subplots(nrows=1, figsize=(18, 5))

plot_distribution(consumption, 'cons_12m', axs)


# In[15]:


cs =client_df['forecast_price_pow_off_peak'].mode()[0]
print(cs)


# In[17]:


sns.countplot(x='has_gas', data=client_df)
plt.xlabel('True/False')
plt.ylabel('Gas')
plt.title('Is it a gas client or not')
plt.show()


# In[ ]:


plt.scatter(client_df['id'], client_df['forecast_price_energy_peak'])
plt.xlabel('customers')
plt.ylabel('Forecast Price Energy On Peak')
plt.title('Scatter Plot between Off-Peak and On-Peak Prices')
plt.show()


# In[ ]:




