#Sentimental Analysis
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import math
import warnings
import streamlit as st
warnings.filterwarnings('ignore') # Hides warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)
sns.set_style("whitegrid") # Plotting style
np.random.seed(7) # seeding random number generator

st.title('AMAZON PRODUCT REVIEW SENTIMENTAL ANALYSIS')

uploaded_file1 = st.file_uploader("Choose a file")
if uploaded_file1 is not None:
  df = pd.read_csv(uploaded_file1)
  st.write(df)

data = df.copy()
data["Product_id"].unique()

#data visulization
#fig = data.hist(bins=50, figsize=(20,15))
#st.pyplot(fig)

from sklearn.model_selection import StratifiedShuffleSplit
print("Before {}".format(len(data)))
dataAfter = data.dropna(subset=["reviews.rating"])
# Removes all NAN in reviews.rating
print("After {}".format(len(dataAfter)))
dataAfter["reviews.rating"] = dataAfter["reviews.rating"].astype(int)

split = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
for train_index, test_index in split.split(dataAfter, dataAfter["reviews.rating"]):
    strat_train = dataAfter.reindex(train_index)
    strat_test = dataAfter.reindex(test_index)

reviews = strat_train.copy()  

#graph
fig = plt.figure(figsize=(16,10))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex = ax1)
reviews["Product_id"].value_counts().plot(kind="bar", ax=ax1, title="Product_id Frequency")
np.log10(reviews["Product_id"].value_counts()).plot(kind="bar", ax=ax2,title="Product_id Frequency (Log10 Adjusted)")
st.pyplot(fig)

st.write(reviews["reviews.rating"].mean())

asins_count_ix = reviews["Product_id"].value_counts().index
fig1 = plt.figure(figsize = (14,6))

plt.subplot(2,1,1)
reviews["Product_id"].value_counts().plot(kind="bar", title="Product_id Frequency")

plt.subplot(2,1,2)
sns.pointplot(x="Product_id", y="reviews.rating", order=asins_count_ix, data=reviews)
plt.xticks(rotation=90)
st.pyplot(fig1)

def sentiments(rating):
    if (rating == 5) or (rating == 4):
        return "Positive"
    elif rating == 3:
        return "Neutral"
    elif (rating == 2) or (rating == 1):
        return "Negative"
# Add sentiments to the data
strat_train["Sentiment"] = strat_train["reviews.rating"].apply(sentiments)
strat_test["Sentiment"] = strat_test["reviews.rating"].apply(sentiments)
st.write(strat_train["Sentiment"][:20])