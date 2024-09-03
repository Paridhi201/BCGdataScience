#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering
# 
# ---
# 
# 1. Import packages
# 2. Load data
# 3. Feature engineering
# 
# ---
# 
# ## 1. Import packages

# In[1]:


import pandas as pd


# ---
# ## 2. Load data

# In[6]:


df = pd.read_csv(r"C:\Users\parid\Downloads\clean_data_after_eda.csv")
price_df = pd.read_csv(r"C:\Users\parid\Downloads\price_data (1).csv")
df["date_activ"] = pd.to_datetime(df["date_activ"], format='%Y-%m-%d')
df["date_end"] = pd.to_datetime(df["date_end"], format='%Y-%m-%d')
df["date_modif_prod"] = pd.to_datetime(df["date_modif_prod"], format='%Y-%m-%d')
df["date_renewal"] = pd.to_datetime(df["date_renewal"], format='%Y-%m-%d')


# In[3]:


df.head(3)


# ---
# 
# ## 3. Feature engineering
# 
# ### Difference between off-peak prices in December and preceding January
# 
# Below is the code created by your colleague to calculate the feature described above. Use this code to re-create this feature and then think about ways to build on this feature to create features with a higher predictive power.

# In[8]:


price_df["price_date"] = pd.to_datetime(price_df["price_date"], format='%Y-%m-%d')
price_df.head()


# In[5]:


# Group off-peak prices by companies and month
monthly_price_by_id = price_df.groupby(['id', 'price_date']).agg({'price_off_peak_var': 'mean', 'price_off_peak_fix': 'mean'}).reset_index()

# Get january and december prices
jan_prices = monthly_price_by_id.groupby('id').first().reset_index()
dec_prices = monthly_price_by_id.groupby('id').last().reset_index()

# Calculate the difference
diff = pd.merge(dec_prices.rename(columns={'price_off_peak_var': 'dec_1', 'price_off_peak_fix': 'dec_2'}), jan_prices.drop(columns='price_date'), on='id')
diff['offpeak_diff_dec_january_energy'] = diff['dec_1'] - diff['price_off_peak_var']
diff['offpeak_diff_dec_january_power'] = diff['dec_2'] - diff['price_off_peak_fix']
diff = diff[['id', 'offpeak_diff_dec_january_energy','offpeak_diff_dec_january_power']]
diff.head()


# Now it is time to get creative and to conduct some of your own feature engineering! Have fun with it, explore different ideas and try to create as many as yo can!

# In[10]:


df = df.drop(columns=['forecast_discount_energy', 'channel_sales'])


# In[ ]:




