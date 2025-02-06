#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
dataset = pd.read_csv("hate_speech.csv")
dataset.head()


# In[20]:


dataset.shape


# In[21]:


dataset.label.value_counts()


# In[22]:


for index, tweet in enumerate(dataset["tweet"][10:15]):
    print(index+1,"-",tweet)


# In[23]:


import re
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\']', ' ',text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.lower()
    return text


# In[24]:


dataset['clean_text'] = dataset.tweet.apply(lambda x: clean_text(x))


# In[25]:


from nltk.corpus import stopwords
len(stopwords.words('english'))


# In[26]:


stop = stopwords.words('english')


# In[27]:


def gen_freq(text):
    word_list = []
    for tw_words in text.split():
        word_list.extend(tw_words)
    word_freq = pd.Series(word_list).value_counts()
    word_freq = word_freq.drop(stop, errors='ignore')
    return word_freq


# In[28]:


def any_neg(words):
    for words in words:
        if words in ['n', 'no', 'non', 'not'] or re.search(r"\wn't", words):
            return 1
        else:
            return 0


# In[29]:


def any_rare(words, rare_100):
    for words in words:
        if words in rare_100:
            return 1
        else:
            return 0


# In[30]:


def any_rare(words, rare_100):
    for words in words:
        if words in rare_100:
            return 1
        else:
            return 0


# In[31]:


def is_question(words):
    for word in words:
        if word in ['when', 'what', 'how', 'why', 'who', 'where']:
            return 1
        else:
            return 0


# In[32]:


word_freq = gen_freq(dataset.clean_text.str)

rare_100 = word_freq[-100:]

dataset['word_count'] = dataset.clean_text.str.split().apply(lambda x: len(x))
dataset['any_neg'] = dataset.clean_text.str.split().apply(lambda x: any_neg(x))
dataset['is_question'] = dataset.clean_text.str.split().apply(lambda x: is_question(x))
dataset['any_rare'] = dataset.clean_text.str.split().apply(lambda x: any_rare(x, rare_100))
dataset['char_count'] = dataset.clean_text.str.split().apply(lambda x : len(x))


# In[33]:


dataset.head(10)


# In[41]:


from sklearn.model_selection import train_test_split

X = dataset[['word_count', 'any_neg', 'any_rare', 'char_count', 'is_question']]
Y= dataset.label
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)


# In[42]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model = model.fit(X_train, Y_train)
pred = model.predict(X_test)


# In[43]:


model.predict(X_test[5:10])


# In[44]:


from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(Y_test, pred)*100, "%")


# In[46]:


from sklearn.ensemble import RandomForestClassifier
clf_rf=RandomForestClassifier()
clf_rf.fit(X_train,Y_train)
rf_pred=clf_rf.predict(X_test).astype(int)


# In[47]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(Y_test,rf_pred))
print(classification_report(Y_test,rf_pred))
print("Accuracy:",accuracy_score(Y_test, rf_pred))


# In[48]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(X_train, Y_train)


# In[49]:


Y_pred = logreg.predict(X_test)


# In[50]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))


# In[ ]:




