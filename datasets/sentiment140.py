from keras.utils.data_utils import get_file
import os, csv, subprocess
import zipfile
import numpy as np
import pandas as pd
from util import preprocess
from io import StringIO


def load_data():
    train_ratio = 0.7

    # path = get_file('sentiment.zip',
    #                  origin='http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip')
    
    # with zipfile.ZipFile(path) as zip:
    #     dir = os.path.join(os.path.expanduser('~'), '.keras/datasets/sentiment')
    #     filename = zip.namelist()[1]
    #     path = zip.extract(filename, dir)
        
    #     subprocess.call('nkf --overwrite --oc=UTF-8-BOM %s' % path)

        # filename = zip.namelist()[0]  # 498件のデータ
        # file = zip.open(filename)
        # file.write('\ufeff')

        # with open(path) as file:
        #     file.read()
        #     # file.write('\ufeff')
        #     # cols = ['sentiment','id','date','query_string','user','text']
        #     # df = pd.read_csv(file, header=None, names=cols, error_bad_lines=False, encoding="latin1")
        #     print(df)
      
        # df = pd.read_csv(file, header=None, names=cols)
    cols = ['sentiment','id','date','query_string','user','text']
    df = pd.read_csv('~/.keras/datasets/sentiment/training.1600000.processed.noemoticon.csv', header=None, names=cols)
    # df = pd.read_csv('~/.keras/datasets/sentiment/testdata.manual.2009.06.14.csv', header=None, names=cols)
    df = df.sample(frac=1, random_state=0)

    # コーパスを作成
    all_text = ' '.join(df['text'])
    corpus, word_to_id, id_to_word = preprocess(all_text)
    # print('corpus: ', corpus)

    # テキストを単語IDに変換する
    df['text'] = df['text'].apply(lambda t: text2word_id_list(t, word_to_id))

    # データを学習用とテスト用に分割する    
    p = int(train_ratio * len(df))
    train_data = df.iloc[:p, :]
    test_data = df.iloc[p:, :]

    return (train_data['text'], train_data['sentiment']), (test_data['text'], test_data['sentiment'])


def text2word_id_list(text, word_to_id):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')
    return [word_to_id[w] for w in words]
