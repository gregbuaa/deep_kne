from model.sketch import CountSketch,compute_tensorsketch_from_cs
import numpy as np
import math

def sample_gumbel(props):
    '''
    sample from the discrete props by gumbel-max trick.
    :param props: unnorm probability
    :return:
    '''
    props = np.array(props) / sum(props)
    logits = np.log(props)
    noise = np.random.gumbel(size=len(logits))
    sample = np.argmax(logits+noise)
    return sample

#### using numpy to compute the kgram vectors.
class kernel_class(object):
    def __init__(self, k, max_p,  nr_tables, table_size, random_files, has_classes = False, has_community=False, max_value=200000):
        self.k = k
        self.max_p = max_p
        self.nr_tables = nr_tables

        self.max_value = max_value
        self.table_size = table_size
        self.random_files = random_files
        self.has_community = has_community
        self.has_classes = has_classes

        self.labels_map = {}
        print('Count sketch data structures initialization')
        self.cs = CountSketch(self.table_size, self.nr_tables * self.max_p, self.random_files,max_value=max_value)

        self.p = 1
        self.q = 1
        self.length = 10
        self.type = 'node2vec'
        self.h = 2


    def set_sample_type(self, type="h",p=1,q=1,length=10):
        if type == 'h':
            self.h = p
        elif type== "node2vec":
            self.p = p
            self.q = q
            self.length = length

        self.type = type


    def generate_feature_maps(self, kernels, graph):
        Gs = [graph['E'], graph['V']]
        if self.has_classes:
            Gs.append(graph['class'])
        if self.has_community:
            Gs.append( graph['community'])

        newGs = []
        for i in range(len(Gs)):
            newG = {}
            for key, value in Gs[i].items():
                newG[int(key)] = value
            newGs.append(newG)
        del Gs

        E, V  = newGs[0], newGs[1]
        community = None 
        classes = None
        if self.has_classes:
            classes = newGs[2]
        
        if self.has_community:
            community = newGs[-1]

        print("Graph kernel starting ...")
        feature_map, graph_map = kernels.onegraph2map(V, E, classes, community)

        return feature_map, graph_map


    def sketch_polynomial_feature_map(self, label_vector, cosine):
        self.cs.clear()

        if cosine:
            norm2 = 0
            for key, value in label_vector.items():
                norm2 += value * value
            if norm2 == 0:
                norm2 = 1
            norm2 = math.sqrt(norm2)

            for key, value in label_vector.items():
                label_vector[key] = value / norm2
            # label_vector = [x / norm2 for x in label_vector]

        self.cs.add_dict(label_vector)
        maps = compute_tensorsketch_from_cs(self.cs, self.max_p, self.nr_tables)

        return maps

    def get_content_kgram_vector(self, words,  k):
        '''
        compute the k-gram
        :param words: word sequence
        :return:
        '''
        if k >= 1:
            local_labels = {}
            sublists = [str(words[i:i + g]) for g in range(1, k + 1) for i in range(len(words) - g + 1)]
            for kgram in sublists:
                local_labels[kgram] = local_labels.get(kgram, 0) + 1
        else:
            local_labels = words

        pc = len(self.labels_map)
        kgram_vector = {}
        for label in local_labels:
            if label not in self.labels_map:
                self.labels_map[label] = pc
                pc += 1
            kgram_vector[self.labels_map[label]] = local_labels[label]

        return kgram_vector

    @staticmethod
    def dict_update(dict1, dict2):
        for key, value in dict2.items():
            dict1[key] = dict1.get(key, 0) + value

        return dict1

    @staticmethod
    def BFS_neighbor( v, V, E):
        '''
        breath-first search neighborhood traversal
        :param v:
        :param V:
        :param E:
        :return:
        '''
        N_v = [v]
        if v in E:
            N_v += E[v]

        return N_v

    @staticmethod
    def BFS_neighbor_h( v, V, E, h):
        N_v = [[] for i in range(h)]
        if v in E:
            N_v[0] = E[v]

        for i in range(h - 1):
            for u in N_v[i]:
                if u not in E:
                    continue
                N_v[i + 1] = N_v[i] + kernel_class.BFS_neighbor(u, V, E)

        return N_v[-1]

    @staticmethod
    def node2vec_neighbor(v, V, E, p, q, length):
        N_v = [v]
        if v not in E:
            return N_v

        curr_neigh = E[v]
        if len(curr_neigh) == 0:
            return N_v + [v] * length

        props = [1 / q] * len(curr_neigh)
        next = curr_neigh[sample_gumbel(props)]
        N_v.append(next)

        for iter in range(length):
            curr = N_v[-1]
            curr_neigh = E[curr]
            pre = N_v[-2]
            pre_neigh = E[pre]
            props = []
            if len(curr_neigh) == 0:
                N_v.append(pre)
                continue
            for next in curr_neigh:
                if next == pre:
                    props.append(1 / p)
                elif next in pre_neigh:
                    props.append(1)
                else:
                    props.append(1 / q)
            next = curr_neigh[sample_gumbel(props)]
            N_v.append(next)

        return N_v

    def get_neighbor_kgram_vector(self, v, V, E, ct_vectors,  k):

        if self.type == 'node2vec':
            N_v = self.node2vec_neighbor(v,V,E,self.p, self.q, self.length)
        else:
            N_v =self.BFS_neighbor_h(v, V, E, self.h)

        ne_vector = {}
        for u in N_v:
            if ct_vectors.__contains__(u):
                vectors = ct_vectors[u]
            else:
                vectors = self.get_content_kgram_vector(V[u],  self.k)
            ne_vector = self.dict_update(ne_vector, vectors)

        if k > 1:
            for i in range(len(N_v) - 1):
                cu = N_v[i]
                nu = N_v[i + 1]
                u_suffix = V[cu][-(k - 1):]
                u_prefix = V[nu][0:k]
                ne_vector = self.dict_update(ne_vector,self.get_content_kgram_vector(u_suffix + u_prefix,  k))

        return ne_vector, N_v

    def content2map(self,V,E, classes, p, q, length, type = 'node2vec'):
        feature_maps = []

        for v in V:
            vertex_map = {}
            vertex_map['vertex'] = v
            vertex_map['class'] = classes[v]

            ct_vector = self.get_content_kgram_vector(V[v],  self.k)
            ct_map = self.sketch_polynomial_feature_map(ct_vector,  True)
            vertex_map['ct_maps'] = ct_map
            if type == 'node2vec':
                N_v = self.node2vec_neighbor(v,V,E,p,q,length)
            else:
                N_v = self.BFS_neighbor_h(v,V,E,p)

            vertex_map['neighbor'] = N_v

            feature_maps.append(vertex_map)

        return feature_maps

    def attr2map(self, content, k):

        ct_vector = self.get_content_kgram_vector(content, k)
        ct_map = self.sketch_polynomial_feature_map(ct_vector, True)
        # ct_tables = compute_countsketch(self.cs, ct_vector, self.max_p, self.nr_tables)
        return ct_map


    def newnode2map(self, teV, trV, teE, teClass, teCom, gvectors):
        '''
        for a graph G=(V,E) generate feature maps. New node adds.
        :param V: {index:content}
        :param E: {index:[index1,index2,]}
        :param classes: {index:class}
        :return:
        '''
        print("process for new node embedding.")

        V = trV.copy()
        V.update(teV)
        feature_maps = {}
        graph_maps = []        ### update graph maps based on gmaps.

        ct_vectors = {}
        for v in teV.keys():
            ct_vectors[v] = self.get_content_kgram_vector(teV[v], self.k)

        com_vertices = {}
        for v, com in teCom.items():
            if not com_vertices.__contains__(com):
                com_vertices[com] = []
            com_vertices[com].append(v)

        graph_vectors = gvectors
        for com, vertices in com_vertices.items():
            if not graph_vectors.__contains__(com):
                graph_vectors[com] = {}

            for v in vertices:
                vertex_map = {}
                vertex_map['vertex'] = v
                vertex_map['class'] = teClass[v]
                vertex_map['graph_num'] = teCom[v]
                vertex_map['ct_maps'] = self.sketch_polynomial_feature_map(ct_vectors[v], True)
                nr_vector, N_v = self.get_neighbor_kgram_vector(v, V, teE, ct_vectors, self.k)
                vertex_map['neighbor'] = N_v
                vertex_map['ne_maps'] = self.sketch_polynomial_feature_map(nr_vector, True)
                feature_maps[v] = vertex_map
                graph_vectors[com] = self.dict_update(graph_vectors[com], nr_vector)

        graph_map = {}
        for gl in graph_vectors:
            graph_map[gl] = self.sketch_polynomial_feature_map(graph_vectors[gl], True)

        graph_maps.append(graph_map)
        graph_maps.append(graph_vectors)

        return feature_maps, graph_maps


    def onegraph2map(self, V, E, classes, community):
        '''
        for a graph G=(V,E) generate feature maps by traversing local neighborhoods, generating strings and sketching the k-gram
        frequency distribution for
        :param V: {index:content}
        :param E: {index:[index1,index2,]}
        :param classes: {index:class}
        :return:
        '''
        print("process graph")

        feature_maps = {}  ### vertex feature maps
        graph_maps = []  ### community feature maps

        ct_vectors = {}
        for v in V:
            ct_vectors[v] = self.get_content_kgram_vector(V[v],  self.k)

        index = 0

        com_vertices = {}
        if self.has_community:
            for v, com in community.items():
                if not com_vertices.__contains__(com):
                    com_vertices[com] = []
                com_vertices[com].append(v)
        else:
            ## there is only one community.
            com_vertices={0:V.keys()}

        graph_vectors = {}
        for com, vertices in com_vertices.items():
            graph_vectors[com] = {}
            for v in vertices:
                if index % 500 == 0:
                    print("deal with vertex ", index)
                    print("current feature map ", len(self.labels_map))
                index += 1

                vertex_map = {}
                vertex_map['vertex'] = v
                if self.has_classes:
                    vertex_map['class'] = classes[v]
                if self.has_community:
                    vertex_map['graph_num'] = community[v]
                ct_vector = ct_vectors[v]
                ct_map = self.sketch_polynomial_feature_map(ct_vector, True)
                vertex_map['ct_maps'] = ct_map
                nr_vector, N_v = self.get_neighbor_kgram_vector(v, V, E, ct_vectors, self.k)
                vertex_map['neighbor'] = N_v
                nr_map = self.sketch_polynomial_feature_map(nr_vector, True)
                vertex_map['ne_maps'] = nr_map  # (p, table_size)
                feature_maps[v] = vertex_map
                if self.has_community:
                    graph_vectors[com] = self.dict_update(graph_vectors[com], nr_vector)

        graph_map = {}
        for gl in graph_vectors.keys():
            graph_map[gl] = self.sketch_polynomial_feature_map(graph_vectors[gl], True)

        graph_maps.append(graph_map)
        graph_maps.append(graph_vectors)

        return feature_maps, graph_maps








