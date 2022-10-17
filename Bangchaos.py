# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 21:26:27 2022

@author: bangchao xie
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support

'''定义要导入的数据类别'''
'''
categories = ['talk.politics.misc', 'talk.religion.misc', 'soc.religion.christian', 
              'talk.politics.guns', 'talk.politics.mideast', 'sci.med', 'sci.space',
              'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'rec.motorcycles', 
              'rec.sport.baseball', 'misc.forsale', 'rec.autos', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'comp.windows.x', 'comp.graphics', 'comp.os.ms-windows.misc', 
              'alt.atheism']
'''

#导入原始数据
twenty_train = fetch_20newsgroups(subset='train',shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test',shuffle=True, random_state=42)
'''
#------------------不那么简洁的实现方法：count-NB-----------------------
#原始文字数据转换为count_vect
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

#定义NB分类器
clf = MultinomialNB().fit(X_train_counts, twenty_train.target)

#测试性能
docs_test = count_vect.transform(twenty_test.data)   
predicted = clf.predict(docs_test)  
cr = np.mean(predicted == twenty_test.target)  
print('count-NB')
print(cr)
'''

'''---------------------------count-NB-------------------------------'''
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB(alpha=0.01)),
    ])

text_clf.fit(twenty_train.data, twenty_train.target)
#测试性能
predicted = text_clf.predict(twenty_test.data)
print('count-NB alpha=0.01: precision, recall, F1-score:')
print(precision_recall_fscore_support(twenty_test.target, predicted, average='macro'))

'''----------------------------tf-NB--------------------------------'''
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', MultinomialNB(alpha=0.0001)),
])

text_clf.fit(twenty_train.data, twenty_train.target)

#测试性能
predicted = text_clf.predict(twenty_test.data)
cr = np.mean(predicted == twenty_test.target)  #计算正确率
print('tf-NB alpha=0.0001: precision, recall, F1-score:')
print(precision_recall_fscore_support(twenty_test.target, predicted, average='macro'))

'''-----------------------------tf-idf-NB-------------------------------'''
#建一个Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(use_idf=False)),
    ('clf', MultinomialNB(alpha=0.0001)),
])

text_clf.fit(twenty_train.data, twenty_train.target)

#测试性能
predicted = text_clf.predict(twenty_test.data)  #将测试集数据放到分类器中
cr = np.mean(predicted == twenty_test.target)  #计算正确率
print('tf-itf-NB alpha=0.0001: precision, recall, F1-score:')
print(precision_recall_fscore_support(twenty_test.target, predicted, average='macro'))

