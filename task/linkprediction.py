import sys
sys.path.append("..")

import time

import random
from sklearn.metrics import roc_auc_score
import numpy as np
from model.tools import read_vectors_from_file,write_vectors_to_file,tranform_key_type


def calculate_distance( a, b): # N * emb_size
    sum = np.dot(a,b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return sum/(norm_a * norm_b)

def generate_neg_sample_from_community(current_community, max_vertices, community_raito=1.0):
    string = None
    r = random.random()
    if r >= community_raito:
        start = random.randint(0, max_vertices)
        end = random.randint(0, max_vertices)       
    else:
        start = random.choice(current_community)
        end = random.choice(current_community)

    return str((start,end))


def prepare_sample(graph, train_ratio = 0.4, community_ratio = 0.0):
    ### remove a porption of edges from graph to test the link prediction. 
    E = graph['E']
    E = tranform_key_type(E)
    community_set = {}
    if 'community' not in graph:
        community_ratio = 0.0
    else:
        community = graph['community']
        community = tranform_key_type(community)
        for vertex, value in community.items():
            if value not in community_set:
                community_set[value] = []
            community_set[value].append(vertex)

    max_vertices = len(graph['V'])
    rEs, newEdges = {}, {}
    for key, values in E.items():
        if not rEs.__contains__(key):
            rEs[key] = []

        for value in values:
            r = random.random()
            if r <= train_ratio:
                if value not in rEs[key]:
                    rEs[key].append(value)
                else:
                    continue
                if  not rEs.__contains__(value):
                    rEs[value] = []
                rEs[value].append(key)
            else:
                newEdges[str((key, value))] = 1
                community_info = {}
                if 'community' in graph:
                    community_info = community_set[community[key]]
                neg_sample = generate_neg_sample_from_community(community_info, max_vertices, community_ratio)
                newEdges[neg_sample] = 0

    graph['E'] = rEs
    test_samples = newEdges

    return graph, test_samples


def evaluateROC(embeds, samps):
    pred = []
    y = []
    for keys, label in samps.items():
        v1, v2 = eval(keys)
        v1, v2 = str(v1), str(v2)
        if embeds.__contains__(str(v1)) and embeds.__contains__(v2):
            e1 = np.array(embeds[v1]['embedding'])
            e2 = np.array(embeds[v2]['embedding'])
            pred.append(calculate_distance(e1,e2))
            y.append(label)
    roc = roc_auc_score(y, pred)
    if roc < 0.5:
        roc = 1 - roc
    return roc



def link_prediction(config):
    graph = read_vectors_from_file(config['graph_path']+config['graph_file'])
    graph, test_samples = prepare_sample(graph, train_ratio=config['train_ratio'], community_ratio= config['community_ratio'])
    write_vectors_to_file(graph, config['graph_path']+config["sparse_graph"])
    write_vectors_to_file(test_samples, config['graph_path']+config['test_samples'])

    return graph, test_samples


if __name__=="__main__":
    from train_kde import train,compute_sketch

    config = {}
    config['graph_path'] = "../graphs/cora/"
    config['graph_file'] = "cora5.txt"
    config['train_ratio'] = 0.4
    config["sparse_graph"] = "cora%.1f.txt"%config['train_ratio']
    config['test_samples'] = "cora-edge%.1f.txt"%config['train_ratio']
    config['community_ratio'] = 0.0

    link_prediction(config)

    train_config = {
        "batch_size":3000,
        "table_size":300,
        'nr_tables':2,
        "embedding_size":300,
        "max_p":3,
        "random_files":"../random/",
        "alpha":0.1,
        "kernel_dim":100,
        'epoch':2,
        "model":"kne",
        'fm_path':"../feature_maps/",
        "embed_file":"../embedding/embedding.txt",
        "graph_file":"../graphs/cora/cora0.4.txt",
        "k": 0
    }
    start = time.time()
    compute_sketch(train_config)
    end = time.time()

    print("computing sketch time", end-start,flush=True)

    start = time.time()
    train(train_config)
    end = time.time()
    print("training time", end - start)

    embeddings = read_vectors_from_file(train_config["embed_file"])
    samps = read_vectors_from_file(config['graph_path']+config['test_samples'])

    print(len(embeddings))

    roc = evaluateROC(embeddings, samps)

    print('ROC is, ', roc,flush=True)




    












