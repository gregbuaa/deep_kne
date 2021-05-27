##### generate the sample txt based on node2vec
##### negative samples based on it.
import numpy as np

class generate_samples(object):

    def __init__(self, texts, windows_size, min_count = 0, format = "Multiple"):
        '''

        :param texts: [[1,2,3],[8,2,1],...]
        :param min_count:
        '''
        self.texts = texts
        self.min_count = min_count

        self.txts_length = 0

        self.get_word_frequency(self.min_count)
        self.init_sample_table()
        self.windows_size = windows_size
        if format == "Multiple":
            self.pairs = self.get_all_pairs(windows_size)
        else:
            self.pairs = self.get_single_pairs(windows_size)


    def get_word_frequency(self, min_count):
        word_frequency = {}
        for text in self.texts:
            self.txts_length = len(text)
            for w in text:
                word_frequency[w] = word_frequency.get(w,0) + 1

        self.word_frequency = {}
        for w, c in word_frequency.items():
            if c < min_count:
                self.txts_length -= c
                continue
            self.word_frequency[w] = c

        self.word_count = len(self.word_frequency)

    def init_sample_table(self):
        sample_table = []
        sample_table_size = 1e8

        wids = []
        frequency = []
        for w, c in self.word_frequency.items():
            wids.append(w)
            frequency.append(c)

        pow_frequency = np.array(frequency)**0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio*sample_table_size)
        for index, c in enumerate(count):
            wid = wids[index]
            sample_table += [wid] * int(c)

        self.sample_table = np.array(sample_table)

    def get_single_pairs(self,window_size):
        pairs = []
        for text in self.texts:
            text_len = len(text)
            for i, u in enumerate(text):
                assert u < self.word_count
                if i >= text_len - window_size:
                    break
                N_u = text[max(0, (i -window_size + 1)):i + window_size + 1]
                for v in N_u:
                    if u == v:
                        continue
                    pairs.append((u,[v]))
        return pairs


    def get_all_pairs(self, window_size):
        print("generating the train samples ....")
        pairs = []
        for text in self.texts:
            text_len = len(text)
            for i, u in enumerate(text):
                assert u < self.word_count
                if i >= text_len - window_size:
                    break
                N_u = text[(i+1):i+window_size+1]

                pairs.append((u,N_u))
        print("has done ...")
        return pairs

    def get_neg_sampling(self, batch_size, count):
        neg_v = np.random.choice(
            self.sample_table, size=(batch_size, count))

        return neg_v






