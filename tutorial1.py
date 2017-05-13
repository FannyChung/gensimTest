#!/usr/bin/python
# -*- coding: UTF-8 -*-

import logging#logging可选
from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)#logging可选

#一个语聊，只写了对应文本中有词的向量
corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],[(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],[(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],[(0, 1.0), (4, 2.0), (7, 1.0)],[(3, 1.0), (5, 1.0), (6, 1.0)],[(9, 1.0)],[(9, 1.0), (10, 1.0)],[(9, 1.0), (10, 1.0), (11, 1.0)],[(8, 1.0), (10, 1.0), (11, 1.0)]]

#transformation：把向量化的文档转化成另一种形式
#这里把词袋向量转化成TFIDF权重向量：忽略常见项，并且向量长度为单位长度
tfidf = models.TfidfModel(corpus)#得到的TFIDF有28个非0元素
print(tfidf)
vec = [(0, 1), (4, 1)]#需要查询的向量
print(tfidf[vec])#把这个向量转化成TFIDF向量，权重结果为(0,1)

#相似度查询
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)
sims = index[tfidf[vec]]#计算需要查询的向量和语料中每个文档的相似度（要用权重向量作为输入）
print(sims)
print(list(enumerate(sims)))