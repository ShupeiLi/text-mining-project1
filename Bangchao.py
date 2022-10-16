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


'''定义要导入的数据类别'''

categories = ['talk.politics.misc', 'talk.religion.misc', 'soc.religion.christian', 
              'talk.politics.guns', 'talk.politics.mideast', 'sci.med', 'sci.space',
              'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'rec.motorcycles', 
              'rec.sport.baseball', 'misc.forsale', 'rec.autos', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'comp.windows.x', 'comp.graphics', 'comp.os.ms-windows.misc', 
              'alt.atheism']


#导入原始数据
twenty_train = fetch_20newsgroups(subset='train',categories=categories, 
                                  shuffle=True, random_state=42)


#原始文字数据转换为count_vect
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

#建立特征索引字典
count_vect.vocabulary_.get(u'algorithm')


'''
count-NB
'''

#转换为TF矩阵

#tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
#X_train_tf = tf_transformer.transform(X_train_counts)

#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#定义NB分类器
clf = MultinomialNB().fit(X_train_counts, twenty_train.target)

#建一个Pipeline

#text_clf = Pipeline([
#    ('vect', CountVectorizer()),
#    ('clf', MultinomialNB()),
#])

#训练
text_clf.fit(twenty_train.data, twenty_train.target)

#在测试集上评估分类器性能
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42) #获取测试集数据，放到twenty_test里面
docs_test = twenty_test.data 
predicted = S(docs_test)  #将测试集数据放到分类器中
cr = np.mean(predicted == twenty_test.target)  #计算正确率
print('count-NB')
print(cr)




'''
tf-NB
'''
#转换为TF矩阵

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


#定义NB分类器

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

  
  #建一个Pipeline

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

#训练
text_clf.fit(twenty_train.data, twenty_train.target)

#在测试集上评估分类器性能

twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42) #获取测试集数据，放到twenty_test里面
docs_test = twenty_test.data 
predicted = text_clf.predict(docs_test)  #将测试集数据放到分类器中
cr = np.mean(predicted == twenty_test.target)  #计算正确率
print('tf-NB')
print(cr)


'''
tf-idf-NB
'''
#计算原始数据TF

tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

#计算原始数据TF-IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


#使用提供数据训练分类器，基于NB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

  
  #建一个Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

#训练
text_clf.fit(twenty_train.data, twenty_train.target)

#在测试集上评估分类器性能
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42) #获取测试集数据，放到twenty_test里面
docs_test = twenty_test.data 
predicted = text_clf.predict(docs_test)  #将测试集数据放到分类器中
cr = np.mean(predicted == twenty_test.target)  #计算正确率
print('tf-idf-NB')
print(cr)


