import os
import numpy as np
import pandas as pd
from utils.tokenizer import segment

BASE_DIR= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REMOVE_WORDS=['|','[',']','语音','图片',' ']

def read_stopwords(path):
    """
    按行读取文件，微处理（把俩边的空格去掉），保存在一个set中
    :param path:
    :return:
    """
    lines =set()
    with open(path,mode='r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            lines.add(line)
    return lines

def remove_words(words_list):
    """
    将单词按照list读取，去除其中的REMOVE_WORDS
    :param words_list:
    :return:
    """
    words_list =[word for word in words_list if word not in REMOVE_WORDS]
    return words_list

def parse_data(train_path,test_path):
    """
    准备训练集和测试集
    将数据集读入，然后拼接出x 和 y(Series)
    调用了precess_sentence,处理每一句再封装成Series(里边是str)
    将处理过的数据集输出到新的文本（去了俩边空，分词，去不要的）
    :param train_path:
    :param test_path:
    :return:
    """
    train_df = pd.read_csv(train_path,encoding='utf-8')
    train_df.dropna(subset=['Report'],how='any',inplace=True)
    train_df.fillna('',inplace=True) #应该是没有用了这句
    #.str <class 'pandas.core.series.Series'> to <class 'pandas.core.strings.StringMethods'>
    #train_x 的类型是Series,其中一系列的str
    train_x=train_df.Question.str.cat(train_df.Dialogue)
    print('train_x is',len(train_x))
    train_x=train_x.apply(preprocess_sentence)
    print('train_x is',len(train_x))
    train_y=train_df.Report
    print('train_y is',len(train_y))
    train_y=train_y.apply(preprocess_sentence)
    print('train_y is',len(train_y))

    test_df = pd.read_csv(test_path,encoding='utf-8')
    test_df.fillna('',inplace=True)
    test_x=test_df.Question.str.cat(test_df.Dialogue)
    test_x=test_x.apply(preprocess_sentence)
    print('test_x is',len(test_x))
    test_y=[]
    train_x.to_csv('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR),index=None,header=False)
    train_y.to_csv('{}/datasets/train_set.seg_y.txt'.format(BASE_DIR),index=None,header=False)
    test_x.to_csv('{}/datasets/test_set.seg_x.txt'.format(BASE_DIR),index=None,header=False)



def preprocess_sentence(sentence):
    """
    预处理三步（按一个一个的句子读入）：
    将句子切词
    去掉要去掉的词
    把list 重新按空格 封装成string
    :param sentence:
    :return:
    """
    seg_list = segment(sentence.strip(),cut_type='word')
    seg_list = remove_words(seg_list)
    seg_line = ' '.join(seg_list)
    return seg_line

if __name__ == '__main__':
    parse_data('{}/datasets/AutoMaster_TrainSet.csv'.format(BASE_DIR),
               '{}/datasets/AutoMaster_TestSet.csv'.format(BASE_DIR))