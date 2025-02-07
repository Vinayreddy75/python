#!/usr/bin/env python
# coding: utf-8

# In[9]:


text = "I am learning NLP"


# In[10]:


import pandas as pd
pd.get_dummies(text.split())


# In[11]:


text = ["i love NLP and i will learn NlP in 2month"]


# In[12]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(text)
vector = vectorizer.transform(text)


# In[13]:


print(vectorizer.vocabulary_)
print(vector.toarray())


# In[14]:


print(vector)


# In[18]:


df = pd.DataFrame(data=vector.toarray(), columns=vectorizer.get_feature_names() )
df


# In[26]:


get_ipython().system('pip install TextBlob')


# In[16]:


text = "I am lerning NLP"


# In[17]:


import nltk
nltk.download("punkt")

nltk.download("punkt_tab")


# In[18]:


from textblob import TextBlob
TextBlob(text).ngrams(1)


# In[20]:


TextBlob(text).ngrams(2)


# In[21]:


TextBlob(text).ngrams(3)


# In[22]:


TextBlob(text).ngrams(4)


# In[23]:


Text = ["The quick brown fox jumped over the lazy dog.","The dog.", "The fox"]


# In[27]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(Text)
print(vectorizer.vocabulary)
print(vectorizer.idf_)


# In[ ]:


from nltk.stem.porter import porter

