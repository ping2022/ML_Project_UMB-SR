import numpy as np
import torch
import datetime
import math
from tqdm import tqdm
import math
from torch.nn import Module,Parameter
from torch import  nn
import torch.nn.functional as F

OPEN_FILE_1 = False
OPEN_FILE_2 = False
OPEN_FILE_3 = False


class MKM_DATA():
    def __init__(self,data):
        self.data_paddings,self.data_operation_paddings, self.data_masks, self.data_targets = np.array(data[0]),np.array(data[1]), np.array(data[2]), np.array(data[3])
        """
        f1 = open("./model/data_padding.txt", "w")
        f2 = open("./model/data_operation_padding.txt","w")
        f3 = open("./model/data_masks.txt","w")
        f4 = open("./model/data_target.txt", "w")
        for row in self.data_targets:
            f4.write(str(row) + "\n")
        f4.close()
        #print(type(self.data_paddings[0]))

        #print(self.data_paddings.shape)
        for row in self.data_operation_paddings:
            f2.write(str(row))
        f2.close()
        for row1 in self.data_paddings:
            f1.write(str(row1))
        f1.close()
        for row2 in self.data_masks:
            f3.write(str(row2))
        f3.close()
        #print("data_paddings size:", self.data_paddings.shape)
        """

class GNN(Module):
    def __init__(self,hidden_size,step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size*2
        self.gate_size = 3*hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size,self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size,self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size,self.hidden_size,bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size,self.hidden_size,bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size,self.hidden_size,bias=True)

    def GNN_cell(self,A,hidden):
        #print("hidden shape", hidden.shape)
        # hidden shape is (128, max_n_node, 100)
        input_in = torch.matmul(A[:,:,:A.shape[1]],self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:,:,A.shape[1]:2*A.shape[1]],self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in,input_out],2)
        g_i = F.linear(inputs,self.w_ih,self.b_ih) # batch_size * xx * gate_size
        g_h = F.linear(hidden,self.w_hh,self.b_hh)
        i_r,i_i,i_n = g_i.chunk(3,2) # tensors,chunks,dim
        h_r,h_i,h_n = g_h.chunk(3,2)
        resetgate = torch.sigmoid(i_r+h_r)
        inputgate = torch.sigmoid(i_i+h_i)
        newgate = torch.tanh(i_n + resetgate*h_n)
        hy = newgate + inputgate*(hidden-newgate)
        return hy

    def forward(self,A,hidden):
        for i in range(self.step):
            hidden = self.GNN_cell(A,hidden)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def generate_batch_slices(len_data,shuffle=True,batch_size=128): #padding,masks,targets
    #len_data = 20993
    #n_batch = 20993/128 ~ 165
    n_batch = math.ceil(len_data / batch_size)
    #shuffle_args is of shape (21120,)
    shuffle_args = np.arange(n_batch*batch_size)
    if shuffle:
        np.random.shuffle(shuffle_args)
    #slices is of shape (165, 128)
    #slices if type list. but each individual element in slices is type ndarray
    slices = np.split(shuffle_args,n_batch)

    # for each item i in slices (note item i is type ndarray of size 128), check if each
    # individual elemnt within item i is less than len_data. If it is, then keep the element
    # in that item, if it is not then remove that item
    # For example, say n_array = np.array([1,2,3,4,5]) and num = 3
    # then n_array[n_array<num] will return [1,2]
    # I think this helps explain why we say 126/7/8 inconsistency
    # slices is still type list with length 165
    slices = [i[i<len_data] for i in slices]
    return slices


def get_slice(slice_index,data_paddings,data_masks,data_targets):
    # slice_index seems to act as a random access index here. My understanding is that individual
    # elements in slice_index will be used to access individual sequence in data_paddings, data_masks and data_targets
    # so inputs, masks and targets will each be type ndarray, with shape of (len(slice_index), 29). or (slice_index,) for target
    inputs,masks,targets = data_paddings[slice_index],data_masks[slice_index],data_targets[slice_index]
    #print(inputs.shape, masks.shape, targets.shape)
    
    # create four list here
    items,n_node,A,alias_input = [],[],[],[]
    # u_input will be shape (29,) vector, iterating slice_index length 
    for u_input in inputs:
        # n_node individual element contains how many unique u_input a given input sequence have
        # n_node will be length slice_index
        n_node.append(len(np.unique(u_input))) #the length of unique items
    # print(len(n_node))
    # max_n_node contains the maximum number of unique input in a given inputs, this number should not be greater than 29
    max_n_node = np.max(n_node) #the longest unique item length
    # print(max_n_node)
    # iterate through inputs again, not sure why u_mask is here
    for u_input,u_mask in zip(inputs,masks):
        # node is a ndarray contains the unique itme for a given u_input
        # node shape should be (n,) where n is less than 29, u_input shape should be (29,)
        node = np.unique(u_input) # the unique items of inputs, note this is sorted
        # print(node)
        # print(u_input.shape, node.shape)

        # output node(ndarray) to a list, then append a number of 0 to this list where it is capped at the max_n_node of this given slices
        # length should not be greater than 29
        items.append(node.tolist()+(max_n_node-len(node))*[0]) #items list

        #u_A is a ndarray of size (max_n_node. max_n_node)
        u_A = np.zeros((max_n_node,max_n_node))
        # iterate 29 times
        for i in range(len(u_input)-1):
            # only look for the unique items, if the next item is 0 then break out of this loop
            if u_input[i+1] == 0:
                break

            # u contains the "unique index" of the current item in the input, v contains the "unique index" of the next item in the input
            u = np.where(node == u_input[i])[0][0] #np.where return a tuple,so need use [0][0] to show the value
            v = np.where(node == u_input[i+1])[0][0]
            # u_A should be the adjaceny matrix of u and v
            u_A[u][v] +=1
        # add all U component items together, where it will return a vector that contains the occurence where a given v matches with any u
        u_sum_in = np.sum(u_A,0) # in degree
        # for any v that has no occurence with any given u, set its value to 1
        u_sum_in[np.where(u_sum_in == 0)] = 1
        # divide each uv interaction value with the total occurence
        u_A_in = np.divide(u_A,u_sum_in)
        u_sum_out = np.sum(u_A,1) #out degree
        u_sum_out[np.where(u_sum_out ==0)] = 1
        u_A_out = np.divide(u_A.T,u_sum_out)
        # this will be a combination of two adjaency matrix
        u_A = np.concatenate([u_A_in,u_A_out]).T
        A.append(u_A)
        # alisas_input contains the unique index for each item in a given u_input, therefore all elements in alias_input will be less than 29
        alias_input.append([np.where(node == i)[0][0] for i in u_input] )
        #print(alias_input)
    # return alias input - unique index for each item in a given u_input, less than 29
    # A is the weighted graph fed into GNN
    # items contains the unique items sequence
    # masks is still masks
    # targets still targets
    return alias_input,A,items,masks,targets


def get_mkm_slice(slice_index,data_paddings,data_operation_paddings,data_masks,data_targets):
    alias_input, A, items, masks, targets = get_slice(slice_index,data_paddings,data_masks,data_targets)
    operation_inputs = data_operation_paddings[slice_index]
    return alias_input,A,items,operation_inputs,masks,targets

######################################################################
# @parameter
#  input: model: MKM_SR class
#  slice_index: ndarray type, randomly generated, size 126/7/8
#  data: train_data, contains, data_padding, data_operation_padding, data_masks, data_targets
#  itemindexTensor: a tensor for item index
######################################################################
def forward_mkm_model(model,slice_index,data,itemindexTensor):
    alias_inputs,A,items,operation_inputs,masks,targets = get_mkm_slice(slice_index, data.data_paddings,data.data_operation_paddings, data.data_masks, data.data_targets)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    operation_inputs = trans_to_cuda(torch.Tensor(operation_inputs).long())
    #masks batch size 128?127?, feature size 29
    #print("masks.shape", masks.shape)
    #print("slice index:",len(slice_index))
    masks = trans_to_cuda(torch.Tensor(masks).long())

    # entity_hidden is type tensor, shape (batch_size, max_n_node of a given input slices, 100)
    # reminder, max_n_node is obtained by looking at the max number of unique items in a given input slices
    # entity_hidden contains the embedding for each element in a given input sequence
    # relation_hidden contains the embedding for the operation performed in a given input sequence
    # the embedding is of size 100
    #print(operation_inputs.shape)
    entity_hidden,relation_hidden = model.forward(items, A,operation_inputs)
    #print(entity_hidden.shape, relation_hidden.shape)

    # this get function takes the input i and grab the embedding of a given input sequence
    # entity_hidden (125, 25, 100)
    # alias_inputs (125, 29) matrix, contain each unique item index rate
    get = lambda i: entity_hidden[i][alias_inputs[i]]
    #print(entity_hidden[0][0])

    #stack two arrays together. This step is to concat the entity_hidden embedding with the relation_hidden embedding
    seq_hiddens = torch.stack(
        [get(i) for i in torch.arange(len(alias_inputs)).long()])  # batch_size*L-length*hidden_size # todo
    #print(seq_hiddens.shape, entity_hidden.shape)
    seq_hiddens = torch.cat([seq_hiddens,relation_hidden],dim=2)
    #print(seq_hiddens.shape)

    #print(itemindexTensor.shape)
    state = model.predict(seq_hiddens, masks, itemindexTensor)
    return targets, state, masks


def train_predict_mkm(model,train_data,test_data,item_ids,itemid2index, epoch):
    #itemindexTensor.size = 12195. A 12195 sized vector
    itemindexTensor = torch.Tensor(item_ids).long()
    total_loss = 0.0
    #slices is type list, size (165,), a 165 dimension vector
    slices = generate_batch_slices(len(train_data.data_paddings), shuffle=True, batch_size=model.batch_size)
    #print("slices shape",slices)
    index = 0
    # set model training mode
    model.train()
    # zip(slices, np.arange(len(slices))) return a typle of (slice, index)
    # slice index is of type ndarray, j is type int32
    for slice_index, j in zip(slices, np.arange(len(slices))):
        # set optimizer
        model.optimizer.zero_grad()
        targets, scores, masks = forward_mkm_model(model, slice_index, train_data, itemindexTensor)
        # targets 
        targets = [itemid2index[tar] for tar in targets]
        targets = trans_to_cuda(torch.Tensor(targets).long())
        #print(sum)
        # this loss function combines the log_softmax and null loss
        # the scores here is of shape (batch_size, total_entity_item) and targets is of shape (batchsize,)
        # the loss function will convert scores to a softmax output before feeding it into the loss function
        loss = model.loss_function(scores, targets)
        index += 1
        # this is where back propagation happens
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % 100 == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()),datetime.datetime.now())
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = generate_batch_slices(len(test_data.data_paddings), shuffle=False, batch_size=model.batch_size)
    # slice_index is of shape (batch_size,)
    for slice_index in slices:
        # targets if of shape (batch_size,), score is of shape (batch_size, num_item)
        targets, scores, masks = forward_mkm_model(model, slice_index, test_data, itemindexTensor)
        sub_scores = scores.topk(20)[1]  # tensor has the top_k functions   # subscore is of shape (batch_size, 20)
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = [itemid2index[tar] for tar in targets]    # target to index conversion

        # iterate batch_size time, score will be of size (20,), target will be a scalar integer
        for score, target, mask in zip(sub_scores, targets, masks):
            # check if target is in score
            hit.append(np.isin(target, score))
            # check where target is in score
            if len(np.where(score == target)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target)[0][0] + 1))
        
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
  
    return hit, mrr