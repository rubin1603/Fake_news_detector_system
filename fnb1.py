
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split

df=pd.read_csv('cleaned.csv')
df.head()


df_x=df["Message"]
df_y=df["label"]
tfd=TfidfVectorizer(min_df=1, stop_words='english')
x_train, x_test, y_train, y_test = train_test_split(
   df_x, df_y, test_size=0.20, random_state=0)


# In[161]:


x_traincv = tfd.fit_transform(x_train)
a=x_traincv.toarray()
tfd.inverse_transform(a[0])





x_train.iloc[0]



#from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
#clf = RandomForestClassifier(max_depth=2, random_state=0)


# In[170]:


y_train=y_train.fillna(0)
y_train=y_train.astype(int)


# In[171]:


df.info()


# In[174]:


mnb.fit(x_train, y_train)
