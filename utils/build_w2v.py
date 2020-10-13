import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
from utils.data_utils import dump_pkl


BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read_lines(path,col_sep=None):
    """
    将文件按行读取到一个list中
    加了一个是否读取具有某个标志行的，但是这里并无用
    :param path:
    :param col_sep:
    :return:
    """
    lines=[]
    with open(path,mode='r',encoding='utf-8') as f:
        for line in f:
            line =line.strip()
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            else:
                lines.append(line)
    return lines

def extract_sentence(train_x_seg_path,train_y_seg_path,test_seg_path):
    """
    将训练集x 训练集y 和测试集x都放到一个list当中
    :param train_x_seg_path:
    :param train_y_seg_path:
    :param test_seg_path:
    :return:
    """
    ret=[]
    lines = read_lines(train_x_seg_path)
    lines += read_lines(train_y_seg_path)
    lines += read_lines(test_seg_path)
    for line in lines:
        ret.append(line)
    return ret

def save_sentence(lines,sentence_path):
    """
    将Lines中的写入到一个文件中
    :param lines:
    :param sentence_path:
    :return:
    """
    with open(sentence_path,'w',encoding='utf-8') as f:
        for line in lines:
            f.write('%s\n' % line.strip())
    print('save sentence:%s' % sentence_path)

def build(train_x_seg_path,train_y_seg_path,test_seg_path,out_path=None,sentence_path='',
          w2v_bin_path="w2v.bin",min_count=1):
    sentences = extract_sentence(train_x_seg_path,train_y_seg_path,test_seg_path)
    save_sentence(sentences,sentence_path)
    print('train w2v model...')
    #train model
    """
    通过gensim工具完成word2vec的训练，输入格式采用sentence,使用skip-gram,embedding维度为256
    """
    w2v=Word2Vec(sentences=LineSentence(sentence_path),size=256,min_count=min_count,sg=1,workers=8,iter=50)
    # w2v.wv.save_word2vec_format('{}/datasets/self_word2vec.txt'.format(BASE_DIR),binary=False)
    w2v.wv.save_word2vec_format(w2v_bin_path,binary=True)
    print("save %s ok." % w2v_bin_path)
    #test
    sim =w2v.wv.similarity('技师','车主')
    print('技师 vs 车主 similarity score:',sim)

    #load model
    model = KeyedVectors.load_word2vec_format(w2v_bin_path,binary=True)
    word_dict={}
    for word in model.vocab:
        word_dict[word]=model[word]
    dump_pkl(word_dict,out_path,overwrite=True)
    # self_output_file='{}/datasets/utf_word2vec.txt'.format(BASE_DIR)
    # with open(self_output_file,'w',encoding='utf-8') as f:
    #     for word,repution in word_dict.items():
    #         f.write('%s\t%s\n' % (word,repution))


if __name__ == '__main__':
    build('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR),
          '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR),
          '{}/datasets/test_set.seg_x.txt'.format(BASE_DIR),
          out_path='{}/datasets/word2vec.txt'.format(BASE_DIR),
          sentence_path='{}/datasets/sentences.txt'.format(BASE_DIR))