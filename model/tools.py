import json
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re

class Key_feature_extract(object):
    def __init__(self, k, filter_num):
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.cut_model = nltk.WordPunctTokenizer()
        self.k = k
        self.filter_num = filter_num

    def normalize_corpus(self, text):

        nstrs = []
        strs = text.split(' ')
        for str in strs:
            ns = re.sub(r'[^a-zA-Z\s]', '', string=str)
            if ns == str:
                nstrs.append(ns)

        str = ' '.join(nstrs).strip()

        doc = re.sub('[\s]+', ' ', str)
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', string=doc)
        doc = doc.lower()
        doc = doc.strip()
        tokens = self.cut_model.tokenize(doc)
        doc = [token for token in tokens if token not in self.stopwords]
        doc = ' '.join(doc)

        return doc

    def get_key_features(self, corpus, norm=True):
        '''
        corpus: {id1:str1, id2:str2,...}
        '''
        train_text = []
        tid_no = {}  ####  train number --> corpus number
        index = 0
        for cid, text in corpus.items():
            if norm:
                text = self.normalize_corpus(text)
            train_text.append(text)
            tid_no[index] = cid
            index += 1

        vectorizer = CountVectorizer(ngram_range=(1, self.k), stop_words="english")
        transformer = TfidfTransformer()
        tf = vectorizer.fit_transform(train_text)
        tfidf = transformer.fit_transform(tf)
        word = vectorizer.get_feature_names() 

        features = {}
        all_features = set()
        idf_row = tfidf.tocsr()

        for ir, row in enumerate(idf_row):
            dic = row.todok()
            sort_dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)
            # feature = {}
            for index, value in enumerate(sort_dic):
                if index > self.filter_num:
                    break
                feature_id = value[0][1]
                kgram_word = word[feature_id]
                all_features.add(feature_id)

        for ir, row in enumerate(idf_row):
            if ir % 100 == 0:
                print("processing the document ", ir)
            dic = row.todok()
            #             sort_dic= sorted(dic.items(), key=lambda d:d[1], reverse = True)
            feature = {}
            for index, value in dic.items():
                feature_id = index[1]
                frq = tf[ir, feature_id]

                if frq == 0 or feature_id not in all_features:
                    continue
                kgram_word = word[feature_id]
                feature[kgram_word] = 1 #int(frq)

            features[tid_no[ir]] = feature

        return features

def write_vectors_to_file(vectors, filename):
    print("start writing file")
    with open(filename, "w+") as f:
        json.dump(vectors, f)
    print("write the file %s completely..."%(filename))


def read_vectors_from_file(filename):
    print("start reading file",flush=True)
    with open(filename, 'r') as load_f:
        load_dict = json.load(load_f)
    print("reading the file %s completely..." % (filename),flush=True)

    return load_dict


def tranform_key_type(dicts):
    if type(dicts).__name__ != 'dict':
        return dicts
    new_dict = {}
    for key, value in dicts.items():
        new_key = (int)(key)
        new_dict[new_key] = value
    return new_dict