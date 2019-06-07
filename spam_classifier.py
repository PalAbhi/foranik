
# coding: utf-8

# In[3]:


file_path = 'enron1/enron1/ham/0007.1999-12-14.farmer.ham.txt'


# In[4]:


with open(file_path,'r') as infile:
    ham_sample=infile.read()


# In[5]:


print(ham_sample)


# In[6]:


file_path='enron1/enron1/spam/0058.2003-12-21.GP.spam.txt'


# In[7]:


with open(file_path,'r') as f:
    spam_sample=f.read()


# In[8]:


print(spam_sample)


# In[13]:


import glob
import os
emails,labels=[],[]
file_path='enron1/enron1/spam/'
for filename in glob.glob(os.path.join(file_path,'*.txt')):
    with open(filename,'r',encoding="ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(1)


# In[14]:


file_path='enron1/enron1/ham/'
for filename in glob.glob(os.path.join(file_path,'*.txt')):
    with open(filename,'r',encoding="ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(0)


# In[15]:


len(emails)


# In[16]:


len(labels)


# In[17]:


from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
def letters_only(astr):
    return astr.isalpha()


# In[18]:


all_names=set(names.words())


# In[19]:


lemmatizer=WordNetLemmatizer()


# In[20]:


def clean_text(docs):
    cleaned_docs=[]
    for doc in docs:
        cleaned_docs.append(' '.join([lemmatizer.lemmatize(word.lower())
        for word in doc.split()
        if letters_only(word)
        and word not in all_names]))
    return cleaned_docs


# In[21]:


cleaned_emails=[]
cleaned_emails=clean_text(emails)


# In[22]:


cleaned_emails[0]


# In[23]:


from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(stop_words="english",max_features=500)


# In[24]:


term_docs = cv.fit_transform(cleaned_emails)
print(term_docs[0])


# In[25]:


feature_names=cv.get_feature_names()
feature_names[481]


# In[26]:


feature_names[357]


# In[27]:


feature_names[125]


# In[28]:


feature_mapping=cv.vocabulary_


# In[29]:


def get_label_index(labels):
    from collections import defaultdict
    label_index=defaultdict(list)
    for index,label in enumerate(labels):
        label_index[label].append(index)
    return label_index


# In[30]:


label_index=get_label_index(labels)


# In[31]:


label_index


# In[32]:


def get_prior(label_index):
    prior={label:len(index) for label,index in label_index.items() }
    total_count=sum(prior.values())
    for label in prior:
        prior[label]/=float(total_count)
    return prior


# In[33]:


prior=get_prior(label_index)


# In[34]:


prior


# In[35]:


import numpy as np


# In[36]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(cleaned_emails,labels,test_size=0.33, random_state=42)


# In[37]:


len(X_train)


# In[38]:


len(Y_train)


# In[39]:


len(X_test)


# In[40]:


len(Y_test)


# In[41]:


term_docs_train=cv.fit_transform(X_train)


# In[42]:


term_docs_test=cv.transform(X_test)


# In[43]:


from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB(alpha=1,fit_prior=True)
clf.fit(term_docs_train,Y_train)


# In[44]:


pred=clf.predict_proba(term_docs_test)


# In[45]:


pred[0:10]


# In[46]:


prediction=clf.predict(term_docs_test)


# In[47]:


prediction[:10]


# In[48]:


accuracy=clf.score(term_docs_test,Y_test)
print(accuracy*100)


# In[49]:


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test,prediction,labels=[0,1])


# In[50]:


from sklearn.metrics import classification_report
report=classification_report(Y_test,prediction)
print(report)

