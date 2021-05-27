from model.kernel_class import kernel_class
from model.tools import read_vectors_from_file, write_vectors_to_file, tranform_key_type
from model.neg_sample import generate_samples
from model.dataloader import NegDataLoader
from model.kne import CNNTrain
import os.path as osp
import os

import torch
import torch.optim as optim

def compute_sketch(config):
    graph = read_vectors_from_file(config["graph_file"])
    has_communtiy = True if "community" in graph else False
    has_classes = True if "class" in graph else False

    kernels  = kernel_class(config["k"],config['max_p'],config['nr_tables'], config['table_size'], config['random_files'], has_communtiy, has_classes)

    kernels.set_sample_type(type="h", p=2)

    feature_map, graph_map = kernels.generate_feature_maps(kernels, graph)

    write_vectors_to_file(feature_map, config['fm_path']+"/feature_map.txt")
    write_vectors_to_file(graph_map,config['fm_path']+"/graph_map.txt")


def train(config):
    graph = read_vectors_from_file(config["graph_file"])
    feature_maps = read_vectors_from_file(config['fm_path']+"/feature_map.txt")
    graph_maps = read_vectors_from_file(config['fm_path']+"/graph_map.txt")
    feature_maps, graph_maps =  tranform_key_type(feature_maps), tranform_key_type(graph_maps)

    E, V = graph['E'], graph['V']
    classes,community = {}, {}
    if "community" in graph:
        community = graph['community']
    if "class" in graph:
        classes = graph['class']
    V, E, classes, community = tranform_key_type(V),tranform_key_type(E),tranform_key_type(classes),tranform_key_type(community)

    config['vertex_num'] = len(feature_maps)
    config['graph_num'] = len(graph_maps[0])

    config['input_dim'] = config['table_size'] * config['nr_tables']
    output_dim = config['embedding_size']
    max_p = config['max_p']
    kernel_dim = config['kernel_dim']

    config["kernel_size"] = (1,2,3)

    config['dropout_p'] = 0.5

    neg_sample_num = 5
    windows_size = 7
    sample_lens = 9
    walk_iter = 10
    p = 1
    q = 2

    config['neg_sample_num'] = neg_sample_num * windows_size

    #### generate random walks
    print("generating random walks by node2vec")
    walks = []
    for it in range(walk_iter):
        for v in V:
            walks.append(kernel_class.node2vec_neighbor(v,V,E,p,q,sample_lens))

    neg_samples = generate_samples(walks, windows_size)

    loader = NegDataLoader("corpus",feature_maps, graph_maps, neg_samples.pairs, max_ver_num=config['vertex_num'], verbose = True)
    USE_CUDA = torch.cuda.is_available()
    model = CNNTrain(config)

    optimizer = optim.Adam(model.parameters())
    if USE_CUDA:
        model = model.cuda()

    for epoch in range(config['epoch']):
        training_loss,_, _, _ = model.training_model( model, optimizer, loader, neg_samples, USE_CUDA = USE_CUDA, is_training=True,type='tke') 

    
    training_loss,alloutputs, allclasses, vertices = model.training_model( model, optimizer, loader, neg_samples, USE_CUDA = USE_CUDA, is_training=False,type=config['model']) 
    embeddings = model.get_embeddings(alloutputs, allclasses, vertices)
    write_vectors_to_file(embeddings,config['embed_file'])

if __name__=="__main__":
    config = {
        "batch_size":512,
        "table_size":300,
        'nr_tables':2,
        "embedding_size":300,
        "max_p":3,
        "random_files":"random\\",
        "alpha":0.1,
        "kernel_dim":100,
        'epoch':5,
        "model":"kne",
        'fm_path':"feature_maps\\",
        "embed_file":"embedding\\embedding.txt",
        "graph_file":"graphs\\cora\\cora0.4.txt",
        "k": 0
    }

    if not osp.exists(config['fm_path']):
        os.mkdir(config['fm_path'])

    if not osp.exists(config['embed_file']):
        os.mkdir(config["embed_file"])
    
    ## test for computing sketch.
    compute_sketch(config)
    train(config)




















