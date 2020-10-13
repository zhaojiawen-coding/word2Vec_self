import os
from collections import defaultdict
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save_word_dict(vocab,save_path):
    """
    把单词表打印在指定路径
    :param vocab: list[(tuple)]
    :param save_path:
    :return:
    """
    with open(save_path,'w',encoding='utf-8') as f:
        for line in vocab:
            w ,i =line
            f.write("%s\t%d\n" % (w,i))


def read_data(path_1,path_2,path_3):
    """
    将三个文件中的单词都放到了一个list中
    :param path_1:
    :param path_2:
    :param path_3:
    :return:
    """
    with open(path_1,'r',encoding='utf-8') as f1, \
        open(path_2,'r',encoding='utf-8') as f2, \
        open(path_3,'r',encoding='utf-8') as f3:
        words=[]
        for line in f1:
            words = line.split()
        for line in f2:
            words += line.split(' ')
        for line in f3:
            words += line.split(' ')
    return words

def build_vocab(items,sort=True,min_count=0,lower=False):
    """
    建立词典，输出一个是（词，index),一个是（index,词）
    :param items: list[item1,item2...]
    :param sort: 按骗了排序还是按单词原来的顺序排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list[(tuple)]
    """
    result=[]
    if sort:
#         sort by count
        dic=defaultdict(int)
        for item in items:
            # 感觉多此一举
            for i in item.split(" "):
                i=i.strip()
                if not i:continue
                i = i if not lower else item.lower()
                dic[i] += 1
        #sort
        dic=sorted(dic.items(),key=lambda x:x[1],reverse=True)
        #对字典进行enurmate之后，得到的item是一个tuple,(key,value)
        for i ,item in enumerate(dic):
            key=item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
#         sort by items
        for i ,item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)
    """
    建立项目的voacb 和 reverse_vocab,vocab的结构是（词，index),reverse_vocab的结构是（index,词）
    """
    vocab = [(w,i) for i,w in enumerate(result)]
    reverse_vocab = [(w[1],w[0]) for w in vocab]

    return vocab,reverse_vocab


if __name__ == '__main__':
    lines=read_data('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR),
                    '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR),
                    '{}/datasets/test_set.seg_x.txt'.format(BASE_DIR))
    vocab,reverse_vocab = build_vocab(lines)
    save_word_dict(vocab,'{}/datasets/vocab.txt'.format(BASE_DIR))