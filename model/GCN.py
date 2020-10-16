import torch
import torch.nn as nn
import torch.nn.init as init
from model.GAT import GAT
from model.TransformerBlock import TransformerBlock
from .KZWANG import KZWANG


class GCN(NeuralNetwork):

    def __init__(self, config, adj):
        super(GCN, self).__init__()
        self.config = config
        self.uV = adj.shape[0]
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        self.mh_attention = TransformerBlock(input_size=300)
        self.word_embedding = nn.Embedding(V, D, padding_idx=0, _weight=torch.from_numpy(embedding_weights))

        self.relation_embedding = GAT(nfeat=300, uV=self.uV, adj=adj)

        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(600, 300)
        self.fc2 = nn.Linear(300, config['num_classes'])

        self.init_weight()
        print(self)

    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)


    def forward(self, X_tid, X_text):
        X_text = self.word_embedding(X_text) # (N*C, W, D)
        X_text = self.mh_attention(X_text, X_text, X_text)
        X_text = X_text.permute(0, 2, 1)

        rembedding = self.relation_embedding(X_tid)

        conv_block = [rembedding]
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(X_text))
            pool = max_pooling(act)
            pool = torch.squeeze(pool)
            conv_block.append(pool)
        conv_feature = torch.cat(conv_block, dim=1)
        features = self.dropout(conv_feature)

        a1 = self.relu(self.fc1(features))
        d1 = self.dropout(a1)

        output = self.fc2(d1)
        return output
