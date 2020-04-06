#!/usr/bin/env python
# coding: utf-8


import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import tqdm
from gensim.models import FastText, Word2Vec
import multiprocessing


class nmf_list(object):
    def __init__(self,data,by_name,to_list,nmf_n,top_n):
        self.data = data
        self.by_name = by_name
        self.to_list = to_list
        self.nmf_n = nmf_n
        self.top_n = top_n

    def run(self,tf_n):
        df_all = self.data.groupby(self.by_name)[self.to_list].apply(lambda x :'|'.join(x)).reset_index()
        self.data =df_all.copy()

        print('bulid word_fre')
        # 词频的构建
        def word_fre(x):
            word_dict = []
            x = x.split('|')
            docs = []
            for doc in x:
                doc = doc.split()
                docs.append(doc)
                word_dict.extend(doc)
            word_dict = Counter(word_dict)
            new_word_dict = {}
            for key,value in word_dict.items():
                new_word_dict[key] = [value,0]
            del word_dict  
            del x
            for doc in docs:
                doc = Counter(doc)
                for word in doc.keys():
                    new_word_dict[word][1] += 1
            return new_word_dict 
        self.data['word_fre'] = self.data[self.to_list].apply(word_fre)

        print('bulid top_' + str(self.top_n))
        # 设定100个高频词
        def top_100(word_dict):
            return sorted(word_dict.items(),key = lambda x:(x[1][1],x[1][0]),reverse = True)[:self.top_n]
        self.data['top_'+str(self.top_n)] = self.data['word_fre'].apply(top_100)
        def top_100_word(word_list):
            words = []
            for i in word_list:
                i = list(i)
                words.append(i[0])
            return words 
        self.data['top_'+str(self.top_n)+'_word'] = self.data['top_' + str(self.top_n)].apply(top_100_word)
        # print('top_'+str(self.top_n)+'_word的shape')
        print(self.data.shape)

        word_list = []
        for i in self.data['top_'+str(self.top_n)+'_word'].values:
            word_list.extend(i)
        word_list = Counter(word_list)
        word_list = sorted(word_list.items(),key = lambda x:x[1],reverse = True)
        user_fre = []
        for i in word_list:
            i = list(i)
            user_fre.append(i[1]/self.data[self.by_name].nunique())
        stop_words = []
        for i,j in zip(word_list,user_fre):
            if j>0.5:
                i = list(i)
                stop_words.append(i[0])

        print('start title_feature')
        # 讲融合后的taglist当作一句话进行文本处理
        self.data['title_feature'] = self.data[self.to_list].apply(lambda x: x.split('|'))
        self.data['title_feature'] = self.data['title_feature'].apply(lambda line: [w for w in line if w not in stop_words])
        self.data['title_feature'] = self.data['title_feature'].apply(lambda x: ' '.join(x))

        print('start NMF')
        # 使用tfidf对元素进行处理
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(tf_n,tf_n))
        tfidf = tfidf_vectorizer.fit_transform(self.data['title_feature'].values)
        #使用nmf算法，提取文本的主题分布
        text_nmf = NMF(n_components=self.nmf_n).fit_transform(tfidf)


        # 整理并输出文件
        name = [str(tf_n) + self.to_list + '_' +str(x) for x in range(1,self.nmf_n+1)]
        tag_list = pd.DataFrame(text_nmf)
        print(tag_list.shape)
        tag_list.columns = name
        tag_list[self.by_name] = self.data[self.by_name]
        column_name = [self.by_name] + name
        tag_list = tag_list[column_name]
        return tag_list