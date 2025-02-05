#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:





# In[8]:


df = pd.read_csv("tripadvisor_hotel_reviews.csv")
df.head()


# In[9]:


df.info()


# In[32]:


import re
from nltk.corpus import stopwords

def clean(review):
    review = review.lower()
    review = re.sub('[^a-zA-Z0-9\s-]', '', review)  # Improved regex for cleaning
    review = " ".join([word for word in review.split() if word not in stopwords.words('english')])
    return review


# In[33]:


df['Review'][0]


# In[37]:


def corpus(text):
    text_list = text.split()
    return text_list


# In[38]:


df['Review_lists'] = df['Review'].apply(corpus)
df.head()


# In[40]:


print(df.columns)
if 0 <= i < len(df):
    corpus += df['Review lists'].iloc[i]
    print(df.head())
    corpus += df[df.columns[0]].iloc[i]  
else:
    print("Index i is out of range")


# In[41]:


from collections import Counter
mostCommon = Counter(corpus).most_common(10)
mostCommon


# In[42]:


words = []
freq = []
for word, count in mostCommon:
    words.append(word)
    freq.append(count)


# In[7]:


doc_trump = "Mr. Trump became president after winning the political election.though he lost the support of some republican friends, trump is friend with president putin"
doc_election = "oresident trump says putin had no poltical interferce is the election outcome.he says it was a witchhunt by poltical parties.he claimed president putin is a friend who had noting to do with the election"
doc_putin = "post election, vladimir putin became president of russia.president putin had served as the primr minister earlier in his poltical career"
documents = [doc_trump, doc_election, doc_putin]


# In[10]:


from sklearn.feature_extraction.text import Countvectorizer
import pandas as pd 
count_vect = Countvectorizer(stop_wors="english")
count_vect = Countvectorizer()
sparse_matrix = count_vect.fit_transform(documents)
doc_term_matrix = sparse_matrix.todense()
df = pd.Dataframe(doc_term_matrix,columns=count_vect.get_feature_names_out(),index=["doc_trump", "doc_election", "doc_putin"])
df


# In[15]:


get_ipython().system('pip install scikit-leran')


# In[13]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd 
documents = ["Trump is a controversial figure.", 
             "Election results have been disputed.", 
             "Putin holds significant political power."]
count_vect = CountVectorizer(stop_words="english")

# Fit the model and transform the documents into a sparse matrix
sparse_matrix = count_vect.fit_transform(documents)

# Convert the sparse matrix to a dense matrix
doc_term_matrix = sparse_matrix.todense()

# Create a DataFrame with the matrix and feature names
df = pd.DataFrame(doc_term_matrix, columns=count_vect.get_feature_names_out(), 
                  index=["doc_trump", "doc_election", "doc_putin"])

# Show the DataFrame
print(df)


# In[ ]:





# In[2]:


def jaccard_similarity(set1,set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union
set_a = {"language","for","computer","NLP","science"}
set_b = {"NLP","for","language","date","ML","AI"}
similarity = jaccard_similarity(set_a, set_b)
print("jaccard similarity:",similarity)


# In[8]:


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[17]:


responses = [" you can return an item within 7 days of purchase."]


# In[18]:


user_input = "how can i track my order?"


# In[19]:


vectorizer = TfidfVectorizer(stop_words="english")
all_text = responses + [user_input]


# In[22]:


tfidf_matrix = vectorizer.fit_transform(all_text)


# In[24]:


user_vector = tfidf_matrix[-1]
response_vectors = tfidf_matrix[:-1]
cosine_similarity(user_vector, response_vectors)


# In[25]:


most_similar_idx = np.argmax(cosine_similarity)


# In[ ]:


print(f"user query: {user_input}")
print(f"user")


# In[8]:


import io
import random
import string
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer


# In[9]:


import nltk


# In[13]:


nltk.download('popular',quiet=True)
nltk.download("punkt")
nltk.download("wordnet")


# In[18]:


f=open("input.txt","r",errors = "ignore")
raw = f.read()
raw = raw.lower()


# In[ ]:


import nltk
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)


# In[17]:


lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatizer(token) for token in tokens]
remove_punct_dict = dict((ord(punct),None) for punct in string.punctuation)
def lenormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# In[21]:


import random

GREETING_INPUTS = ("hello", "hi", "greetings", "what's up", "hey", "how are you?")
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad you are talking to me", "I am fine! How about you?"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# In[23]:


def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


# In[24]:


flag = True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))
            else:
                print("ROBO: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("ROBO: Bye! take care..")


# In[ ]:




