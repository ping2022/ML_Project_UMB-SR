import torch
from torch import  nn
import torch.nn.functional as F
import math
from torch.nn import Module,Parameter
from model.utils import trans_to_cuda,GNN

class MKM_SR(Module):
    def __init__(self,opt,n_entity,n_relation,n_item):
        super(MKM_SR, self).__init__()
        self.hidden_size = opt.hidden_size
        self.l2 = opt.l2
        self.n_item = n_item
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.batch_size = opt.batch_size
        self.kg_loss_rate = trans_to_cuda(torch.Tensor([opt.kg_loss_rate]).float())

        self.entity_embedding = nn.Embedding(self.n_entity, self.hidden_size)
        self.relation_embedding = nn.Embedding(self.n_relation, self.hidden_size)
        self.norm_vector = nn.Embedding(self.n_relation, self.hidden_size)

        self.gnn_entity = GNN(self.hidden_size, step=opt.step)
        self.gru_relation = nn.GRU(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True)
        self.linear_one = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True)
        self.linear_two = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True)
        self.linear_three = nn.Linear(self.hidden_size * 2, 1, bias=True)
        self.linear_transform = nn.Linear(self.hidden_size * 4, self.hidden_size, bias=True)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.reset_parameters()


    def reset_parameters(self):
        index = 0
        stdv = 1.0 / math.sqrt(self.hidden_size*2)
        for weight in self.parameters():
            index+=1
            weight.data.uniform_(-stdv, stdv)   # initialize data, only called once

    def predict(self, seq_hiddens, masks, itemindexTensor):
        # print([  torch.arange(masks.shape[0]).long(), torch.sum(masks, 1) - 1] )
        #print(masks.shape)
        #print((torch.sum(masks, 1) - 1).shape )
        # ht is of shape (batch_size, 200), where each individual element consists of the column of the last item + operation embeddings.
        ht = seq_hiddens[
            torch.arange(masks.shape[0]).long(), torch.sum(masks, 1) - 1 ]  # the last one #batch_size*hidden_size
        #print(ht.shape)
        # ht is fed into linear_one, output with batch_size,1,hidden_size
        # This is to compute W1ML shown in the paper
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size*1*hidden_size
        
        #print(q1.shape) # q1.shape is (batch_size,1,200)

        # seq_hiddens (item + operation embedding) is fed into linear two.
        # This is to compute W2mt to my understanding
        q2 = self.linear_two(seq_hiddens)  # batch_size*seq_length*hidden_size
        #print(q2.shape)
        #TODO: what is BetaT
        alpha = self.linear_three(torch.sigmoid(q1 + q2))  # batch_size * seq_len *1
        #print(alpha.shape)  # this is shape (batch_size, 29, 1)
        #print(torch.sigmoid(q1+q2).shape)  # this is shape (batch_size, 29, 200)
        # seq_hiddens contain invalid embedding for 0 item id, need masks to cancel out (TODO: need validation)
        a = torch.sum(alpha * seq_hiddens * masks.view(masks.shape[0], -1, 1).float(), 1) # a.shape batch_size *hidden_size
        #print(a.shape, ht.shape) # a has shape (batch_size, 200), ht has shape(batch_size, 200)
        #print(masks.view(masks.shape[0], -1, 1).float().shape)
        #print(seq_hiddens[0])
        a = self.linear_transform(torch.cat([a, ht], 1))
        #print(a.shape)         # a is shape (batch_size, 100)
        # a seems to be the sessions final representation
        # b is the item j to be fed into the MLP
        # Question: what is the structure of the MLP?
        b = self.entity_embedding.weight[itemindexTensor]  # n_items *latent_size
        #print(b)
        #print(b.shape)         # b is shape (num_items, 100)

        # scores is shape (batch_size, num_items)
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A,relation_inputs):
        #print("parent forward")
        entity_hidden = self.entity_embedding(inputs)  # batch,L,hidden_size
        #print(A.shape)
        entity_hidden = self.gnn_entity(A, entity_hidden)  # batch,hidden_size
        relation_inputs = self.relation_embedding(relation_inputs)
        relation_output,relation_hidden = self.gru_relation(relation_inputs,None)
        return entity_hidden,relation_output