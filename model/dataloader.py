import numpy as np

class DataLoader(object):
    def __init__(self, name):
        self.batch_size = 0
        self.ptr = 0
        self.num_batch = None
        self.indexes = None
        self.data_size = None
        self.batch_indexes = None
        self.name = name


    def _shuffle_indexes(self):
        np.random.shuffle(self.indexes)

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self, *args, **kwargs):
        raise NotImplementedError("Have to override prepare batch")

    def epoch_init(self, batch_size, shuffle = True, verbose = True):

        self.ptr = 0
        self.batch_size = batch_size
        self.num_batch = self.data_size // self.batch_size
        if verbose:
            print("Number of left over sample %d" % \
                             (self.data_size - self.batch_size * self.num_batch))

        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])

        if shuffle:
            self._shuffle_batch_indexes()

        if verbose:
            print("%s begins with %d batches" % (self.name, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_index=selected_ids)
        else:
            return None


class NegDataLoader(DataLoader):
    def __init__(self, name, feature_maps, graph_maps, train_samples, max_ver_num=19000, verbose = True):
        super(NegDataLoader, self).__init__(name)

        self.name = name
        self.max_ver_num = max_ver_num
        self.vertex_data =  feature_maps ### All the nodes (vertex_num,)
        self.graph_data = graph_maps[0]
        # self.graph_class = graph_maps[1]

        self.train_samples = train_samples

        self.data_size = len(self.train_samples)

        any_elem = next(iter(self.vertex_data.values()))

        self.p_size = len(any_elem['ct_maps'])
        self.vector_size = len(any_elem['ct_maps'][0])
        self.windows_len = len(self.train_samples[0][1])

        self.indexes = [i for i in range(self.data_size)]

        if verbose:
            print("train sample number:%d, max_p:%d, graphs_num:%d"%(self.data_size, self.p_size, len(self.graph_data)))

    def _prepare_batch(self, selected_index):
        # rows = [self.vertex_data[idx] for idx in selected_index]
        rows = [self.train_samples[idx] for idx in selected_index]

        context = np.zeros([self.batch_size, 3, self.p_size, self.vector_size], dtype=np.float32)
        vertices = np.zeros(self.batch_size, dtype=np.int32)
        classes = np.zeros(self.batch_size, dtype=np.int32)
        communities = np.zeros(self.batch_size, dtype=np.int32)

        neighbors = np.zeros([self.batch_size,self.windows_len], dtype=np.int32)
        neighbors_len = np.zeros(self.batch_size, dtype=np.int32)

        for i in range(len(rows)):
            row = rows[i]
            u = row[0]
            N_u = row[1]
            vertices[i] = u

            current_feature = self.vertex_data[u]
            context[i][0] = current_feature['ct_maps']
            classes[i] = current_feature['class']
            context[i][1] = current_feature['ne_maps']
            graph_num = current_feature['graph_num']
            communities[i] = graph_num
            context[i][2] = self.graph_data[str(graph_num)]
            neighbors[i] = np.array(N_u)
            neighbors_len[i] = len(N_u)


        return {"context":context, "vertex":vertices, "neighbors":neighbors, \
               "class":classes, "community":communities,'neighbor_lens':neighbors_len}


