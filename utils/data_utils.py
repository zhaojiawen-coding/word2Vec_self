import numpy as np
from collections import Counter
import copy
import pickle
import codecs
import chardet
import os
import time

PAD_TOKEN = 'PAD'
GO_TOKEN = 'GO'
EOS_TOKEN = 'EOS'
UNK_TOKEN = 'UNK'

start_id =0
end_id =1
unk_id =2

def save_word_dict(dict_data,save_path):
    """
    读入一个字典，将数据写入一个文件中
    :param dict_data:
    :param save_path:
    :return:
    """
    with open(save_path,'w',encoding='utf-8') as f:
        #dict.items()得到的类型是<class 'dict_items'>，可通过以下方式获取
        for k,v in dict_data.items():
            f.write("%s\t%d\n" % (k,v))

def read_vocab(input_texts,max_size=50000,min_count=5):
    """
    用Counter实现了一个排序，先updata,后most_common(),
    合成了一个单词（可能是是字或字母，全看input_texts的类型）的从多到少的排序的字典
    :param input_texts:
    :param max_size:
    :param min_count:
    :return:dict
    """
    token_counts=Counter()
    special_tokens=[PAD_TOKEN,GO_TOKEN,EOS_TOKEN,UNK_TOKEN]
    for line in input_texts:
        for char in line.strip():
            char =char.strip()
            if not char:
                continue
            token_counts.update(char)
    #sort word count by value
    count_pairs = token_counts.most_common()
    vocab = [k for k,v in count_pairs if v>=min_count]
    #Insert the special tokens to the beginning
    vocab[0:0] =special_tokens
    full_token_id =list(zip(vocab,range(len(vocab))))[:max_size]
    vocab2id = dict(full_token_id)
    return vocab2id

def stat_dict(lines):
    """
    单纯的统计单词的个数
    :param lines:
    :return:
    """
    word_dict={}
    for line in lines:
        tokens =line.split(" ")
        for t in tokens:
            t=t.strip()
            if t:
                word_dict[t] =word_dict.get(t,0) +1#get的第二个参数是default
    return word_dict

def filter_dict(word_dict,min_count=3):
    """
    首先是一个深复制，不会改变原来的word_dict
    然后将复制过的dict中的词频小于某个数的删除
    :param word_dict:
    :param min_count:
    :return:
    """
    out_dict =copy.deepcopy(word_dict)
    for w,c in out_dict.items():
        if c <min_count:
            del out_dict[w]
    return out_dict

def read_lines(path,col_sep=None):
    """
    将文件的每一行读取到list中，其中有个判断词
    :param path:
    :param col_sep:
    :return:
    """
    lines=[]
    with open (path,mode='r',encoding='utf-8') as f:
        for line in f:
            line =line.strip()
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            else:
                lines.append(line)
    return lines

def load_dict(dict_path):
    """
    将文件中的单词读到一个dict中 字典——单词：idx(从0开始）
    :param dict_path:
    :return:
    """
    return dict((line.strip().split("\t")[0],idx)
                for idx,line in enumerate(open(dict_path,'r',encoding='utf-8').readlines()))

def load_reverse_dict(dict_path):
    """
    将文件中的单词读到一个字典中，字典——idx:单词
    :param dict_path:
    :return:
    """
    return dict((idx,line.strip().split("\t")[0])
                for idx,line in enumerate(open(dict_path,mode='r',encoding='utf-8').readlines()))

def flatten_list(nest_list):
    """
    嵌套——递归
    嵌套列表压扁成一个列表
    :param nest_list:
    :return:
    """
    result = []
    for item in nest_list:
        if isinstance(item,list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

def map_item2id(items,vocab,max_len,non_word=0,lower=False):
    """
    将word/pos等映射为id
    相当于有一个词表（word,id) 又给了一个词列表，将词列表对于的id映射出来
    :param items: list,待映射列表
    :param vocab: 词表
    :param max_len: int,序列最大长度
    :param non_word: 未登录词标号，默认为0
    :param lower: 小写
    :return: np.array
    """
    assert type(non_word) == int
    arr = np.zeros((max_len,),dtype='int32')
    #截断max_len长度的items
    min_range = min(max_len,len(items))
    for i in range(min_range):
        item =items[i] if not lower else items[i].lower()
        arr[i] = vocab[item] if item in vocab else non_word
    return arr

def write_vocab(vocab,filename):
    """
    每行一个单词将排好序的单词写到文件中
    这个vocab 应该是 word,index(从0开始）
    :param vocab:
    :param filename:
    :return:
    """
    print("Writteing vocab...")
    with open(filename,'w',encoding='utf-8') as f:
        for word,i in sorted(vocab.items(),key=lambda x:x[1]):
            if i != len(vocab)-1:
                f.write(word + '\n')
            else:
                f.write(word)
    print("-write to {} done . {} tokens".format(filename,len(vocab)))

def load_voacb(filename):
    """
    读取单词，word,index(从0开始）
    :param filename:
    :return:
    """
    try:
        d=dict()
        with open(filename,'r',encoding='utf-8') as f:
            #lines=f.readlines(0
            for idx ,word  in enumerate(f.readlines()):
                word =word.strip()
                d[word] =idx
    except IOError:
        raise IOError(filename)
    return d

def transform_data(data,vocab):
    """
    同样是根据词表的一个映射吧
    :param data:
    :param vocab:
    :return:
    """
    out_data =[]
    for d in data:
        tmp_d = []
        for sent in d:
            tmp_d.append([vocab.get(t,unk_id) for t in sent if t])
        out_data.append(tmp_d)
    return out_data

def load_pkl(pkl_path):
    """
    加载序列化的文件
    加载词典文件
    :param pkl_path:
    :return:
    """
    with open(pkl_path,'rb') as f:
        result= pickle.load(f)
    return result

#这个没用，是自己的记录
def get_content(filename,filename1):
    """
    读取每一行的倒数18个字符，写到另外一个文件中
    :param filename:
    :param filename1:
    :return:
    """
    with open(filename,'rb')as f1:
        #chardet.detect（）返回的是一个字典格式。
        #{'confidence': 1.0, 'language': '', 'encoding': 'UTF-16'}
        charset=chardet.detect(f1.read())
    #查了 codecs.open好像是python2的产物为了方便编码，但是python3的open明明也可以啊
    with codecs.open(filename,'r',chardet['encoding']) as f1,codecs.open(filename1,'w','utf-8') as f2:
        for line in f1.readlines():
            f2.write(line[-18:])

def dump_pkl(vocab,pkl_path,overwrite=True):
    """
    序列化到文件
    :param vocab:
    :param pkl_path:
    :param overwrite:
    :return:
    """
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return
    if pkl_path:
        with open(pkl_path,'wb') as f:
            pickle.dump(vocab,f,protocol=pickle.HIGHEST_PROTOCOL)
        print("save %s ok." % pkl_path)

def get_word_segment_data(contents,word_sep=' ',pos_sep='/'):
    """
    把单词拼接成 word word 如果原来的单词有/就取了/前一部分
    :param contents:
    :param word_sep:
    :param pos_sep:
    :return:
    """
    data=[]
    for content in contents:
        temp=[]
        #将切好的词放到了list中
        for word in content.split(word_sep):
            if pos_sep in word:
                temp.append(word.split(pos_sep)[0])
            else:
                temp.append(word.strip())
        #将切好的词凭借成以空格分开的字符串，再放到新list中
        data.append(word_sep.join(temp))
    return  data

def get_char_segment_data(contents,word_sep=' ',pos_sep='/'):
    """
    将contents中的词 ['d d d',...]
    :param contents:
    :param word_sep:
    :param pos_sep:
    :return:
    """
    data=[]
    for content in contents:
        temp=''
        #把word拼接成一串
        for word in content.split(word_sep):
            if pos_sep in word:
                temp+=word.split(pos_sep)[0]
            else:
                temp+= word.strip()
        #一串处理成以空格分开的字符串（最小单位）
        data.append(word_sep.join(list(temp)))
    return  data

def load_list(path):
    """
    将文件中的数据按空格（不限个数）读到list中
    :param path:
    :return:
    """
    return [word for word in open(path,'r',encoding='utf-8').read().split()]

def save(pred_labels,true_labels=None,pred_save_path=None,data_set=None):
    """
    将pre_label true_lable 和 dataset写到文件中。但是格式为什么这样还是不清楚
    :param pred_labels:
    :param true_labels:
    :param pred_save_path:
    :param data_set:
    :return:
    """
    if pred_save_path:
        with open(pred_save_path,'w',encoding='utf-8') as f:
            for i in range(len(pred_labels)):
                if true_labels and len(true_labels)>0:
                    assert len(true_labels)== len(pred_labels)
                    if data_set:
                        f.write(true_labels[i] + '\t' + data_set[i] + '\n')
                    else:
                        f.write(true_labels[i] + '\n')
                else:
                    if data_set:
                        f.write(pred_labels[i] + '\n' + data_set[i] + '\n')
                    else:
                        f.write(pred_labels[i] + '\n')
        print("pred_save_path:",pred_save_path)

def load_word2vec(params):
    """
    把文件中存储好的word2vec的词向量取出来放到了一个word2vec_matrix
    :param params:
    :return:
    """
    word2vec_dict = load_pkl(params['word2vec_output'])
    vocab_dict = open(params['vocab_path'],encoding='utf-8').readlines()
    embedding_matrix=np.zeros((params['vocab_size'],params['embed_size']))

    for line in vocab_dict[:params['vocab_size']]:
        word_id =line.split()
        word,i=word_id
        embedding_vector=word2vec_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[int(i)] =embedding_vector
    return embedding_matrix

def get_result_filename(params,commit=''):
    """
    拼接了一个输出路径
    :param params:
    :param commit:
    :return:
    """
    save_result_dir =params['test_save_dir']
    batch_size=params['batch_size']
    epochs=params['epochs']
    max_length_inp=['max_dec_len']
    embedding_dim=['embed_size']
    now_time =time.strftime('%Y_%m_%d_%H_%M_%S')
    filename=now_time+'_batch_size_{}_epochs_{}_max_length_inp_{}_embedding_dim{}{}.csv'.format(batch_size,epochs,
                                                                                                max_length_inp,
                                                                                                embedding_dim,
                                                                                                commit)
    result_save_path=os.path.join(save_result_dir,filename)
    return result_save_path