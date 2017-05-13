#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gensim.models import word2vec
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus("huxike_seg.txt")
model = word2vec.Word2Vec(sentences,size=200)

y1 = model.similarity(u"肺炎", u"腹软")
print (u"相似度为：", y1)