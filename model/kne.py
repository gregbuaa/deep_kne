### deep kernel network embedding based on Pytorch.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LoopConv2d(nn.Module):
    def __init__(self, embedding_dim, kernel_dim=100, kernel_size=(1,2,3)):
        super(LoopConv2d, self).__init__()

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_size])
        self.kernel_size = kernel_size

        self.batchnormal = nn.BatchNorm2d(kernel_dim)

    def forward(self, inputs):
        '''
        :param inputs: (B, T, D)
        :return:
        '''
        inputs = inputs.unsqueeze(1) # B, 1, T, D

        outputs = []
        for i in range(len(self.kernel_size)):
            conv = self.convs[i]
            compensate_len = self.kernel_size[i] - 1
            input_k = inputs
            if compensate_len > 0:
                input_k = torch.cat((inputs, inputs[:,:,0:compensate_len,:]),2) ## B 1 T D

            output = F.relu(self.batchnormal(conv(input_k))).squeeze(3) # B, Kd, W
            # output = F.relu(conv(input_k)).squeeze(3)
            output = F.max_pool1d(output, output.size(2)).squeeze(2) # (B, Kd)
            outputs.append(output)

        concated = torch.cat(outputs, 1) ## (B, Kd*len(Ks))

        return concated


class CNNTrain(nn.Module):

    def __init__(self, config):
        super(CNNTrain,self).__init__()

        self.input_dim = config['input_dim']
        self.output_dim = config['embedding_size']
        self.max_p = config['max_p']

        self.alpha = config["alpha"]
        self.neg_sample_num = config["neg_sample_num"]
        self.batch_size = config["batch_size"]

        self.convs = nn.ModuleList([LoopConv2d(self.input_dim, kernel_dim=config['kernel_dim'], kernel_size = config['kernel_size']) for _ in range(self.max_p)])

        self.dropout = nn.Dropout(config['dropout_p'])

        self.fc = nn.Linear(len(config['kernel_size'])*config['kernel_dim']*self.max_p, 2*self.output_dim)

        self.linears = nn.Sequential(
            nn.BatchNorm1d(2*self.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.output_dim, 2 * self.output_dim),
            nn.BatchNorm1d(2*self.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.output_dim, self.output_dim)
        )

        self.cross_loss = nn.CrossEntropyLoss()
        self.v_embedds = nn.Embedding(config['vertex_num'], self.output_dim)
        self.sm3 = nn.Linear(self.output_dim, config['graph_num'])


    def forward(self, inputs, pos_v, neg_v, community, is_training= False):
        '''
        :param inputs: B 3 max_p embedding_dim
        :param pos_v: (B, len(nei))
        :param neg_v: (B, len(nei)*neg_sample)
        :param is_training:
        :return:
        '''
        inputs = inputs.transpose(1,2).transpose(0,1) ## (max_p B 3 embedding_dim)
        inputs = [self.convs[i](inputs[i]) for i in range(self.max_p)] ## (B,Kd*len(Ks))*max_p
        concated = torch.cat(inputs, 1)

        if is_training:
            concated = self.dropout(concated)

        output = self.fc(concated)
        output = self.linears(output) ## B, emb

        loss1, loss2  = 0.0, 0.0
        if is_training:
            emb_v = self.v_embedds(pos_v) ## B,N,emb
            score = torch.bmm(emb_v, output.unsqueeze(2)).squeeze(2) # B,N
            score = F.logsigmoid(score)
            neg_emb = self.v_embedds(neg_v)
            neg_score = torch.bmm(neg_emb, output.unsqueeze(2)).squeeze(2) # B,M
            neg_score = F.logsigmoid(-1*neg_score)
            batch_size, nei_len = pos_v.size(0), pos_v.size(1)
            loss1 = -1 * (torch.sum(score) + torch.sum(neg_score))/(batch_size*nei_len)
            loss2 = self.cross_loss(self.sm3(output), community)

        return output, loss1, loss2


    @staticmethod
    def get_embeddings(outputs, classes, vertices):
        graph_maps = {}
        BATCH_SIZE = np.size(outputs[0],0)
        for i in range(len(outputs)):
            for batch in range(BATCH_SIZE):
                graph = dict()
                graph['embedding'] = outputs[i][batch].tolist()
                graph['class'] = int(classes[i][batch])
                graph_maps[str(vertices[i][batch])] = graph

        return graph_maps

    def training_model(self, model, optimizer, loader, neg_sample,USE_CUDA = False, is_training=False, type='tke-c'):
        outputs, allclasses, vertices = [], [], []

        FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

        loader.epoch_init(self.batch_size)
        batch = loader.next_batch()
        losses = [.01]
        meanloss = [.01]

        index = 0

        while batch is not None:
            inputs = FloatTensor(batch['context']) ## (B 3 max_p embedding_dim)
            if type == 'tke-c':
                inputs = FloatTensor(batch['context'])[:,0:2,:,:]
            classes = batch['class']  # (batch,)
            communities = LongTensor(batch['community'])
            ver = batch['vertex']
            neighbors = LongTensor(batch["neighbors"])  # (batch, max_neighbors_len)
            neg_nei = LongTensor(neg_sample.get_neg_sampling(self.batch_size, self.neg_sample_num))

            if is_training:
                optimizer.zero_grad()

            output, loss1, loss2 = model(inputs, neighbors,neg_nei,communities,is_training=is_training)

            if not is_training:
                outputs.append(output.data.cpu().numpy())
                allclasses.append(classes)
                vertices.append(ver)

            if is_training:
                loss = loss1 + self.alpha * loss2
                losses.append(loss.item())
                meanloss.append(loss.item())

                loss.backward()
                optimizer.step()

            if index % 500 == 0:
                print("[%d/%d] [%d/%d] mean_loss :%0.2f" % (0, 0, index, loader.num_batch, 0.00 if len(losses)==0 else np.mean(losses)),flush=True)
                losses = [.01]

            index += 1
            batch = loader.next_batch()

        return np.mean(meanloss), outputs, allclasses, vertices


