
from __future__ import annotations
from dataclasses import dataclass,field

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

DEBUGE = True
@dataclass
class Node:
    idx: int
    value: Tensor
    adjacency_list: list[Edge] = field(default_factory=list)

@dataclass
class Edge:
    node: Node
    weight: Tensor


# call this function for multi-headed attn
def scaled_graph_attention(query:Tensor, key:Tensor, value:Tensor):
    batch,num_heads,seq_len,head_dim = query.shape
    assert batch==1, "batch size must be one"
    
    # key = key.reshape(batch,num_heads,seq_len,head_dim)
    # query = query.reshape(batch,num_heads,seq_len,head_dim)
    outputs = []
    for head_idx in range(num_heads):
        q = query[:,head_idx,:,:]
        k = key[:,head_idx,:,:]
        v = value[:,head_idx,:,:]

        result = casual_self_attention_with_graph(q,k,v)
        outputs.append(result)
        
    output = torch.stack(outputs,dim=1)
    
    return output.reshape(batch,num_heads,seq_len,head_dim )
    

def casual_self_attention_with_graph(query:Tensor, key:Tensor, value:Tensor):
    batch,seq_len,d_model = query.shape
    nodes = [Node(idx,value[:,idx,:],[]) for idx in range(seq_len)]
    graph = build_graph(nodes,key,query)
    
    # traversing graph
    outputs = []
    for r_idx,root in enumerate(graph):
        curr_value = torch.zeros(1,1,d_model)
        for edge in root.adjacency_list:
            curr_value += edge.node.value * edge.weight
        outputs.append(curr_value)

    output = torch.stack(outputs,dim=-2).squeeze(dim=2)
    
    output = output.reshape(batch,seq_len,d_model)
    
    return output

def build_graph(nodes:list[Node],keys:Tensor,queries:Tensor):    
    batch,seq_len,d_model = queries.shape
    for idx,curr_node in enumerate(nodes):
        # picking 1 to n keys
        keys_history = keys[:,:idx+1,:] 
        
        # picking nth query
        curr_query = queries[:,idx,:]
        
        # here we take dot product (concise similarity) between current query
        # and all keys that contains in histoy of current node (token)
        similarity_values = curr_query@keys_history.transpose(-1,-2)  
        
        # if DEBUGE: print(f"{keys_history.shape=} {curr_query.shape=} {similarity_values.shape=} ")
        similarity_values = similarity_values/math.sqrt(d_model)
        
        # after softmax you will get weights with indicates 
        # how much current node want pay attention to past node
        attn = F.softmax(similarity_values.float(),dim=-1).type_as(keys)
        
        attn = attn.reshape(-1) # reshaping to make it simple
        # if DEBUGE: print(attn)
        
        # adding back edges in adjacency list of each node
        for nidx,node in enumerate(nodes[:idx+1]):
            edge_weight = attn[nidx]
            
            # if DEBUGE: print(f"{idx} attend to {nidx} node with {edge_weight:.2f}")
            edge = Edge(node=node,weight=edge_weight)
            
            # curent node is getting weighted edge with all past nodes
            curr_node.adjacency_list.append(
                edge
            )
    return nodes
   

 
        
@torch.no_grad
def test_attn():
    torch.manual_seed(6)
    batch = 1
    seq_len = 8
    d_model = 2**10
    num_heads = 2
    head_dim = int(d_model/num_heads)

    
    assert batch == 1, "Batch size must be 1 for this test"
    Wk = nn.Linear(d_model, d_model)
    Wq = nn.Linear(d_model, d_model)
    Wv = nn.Linear(d_model, d_model)
    x = torch.rand(batch, seq_len, d_model)
    
    key: Tensor = Wk(x)
    query: Tensor = Wq(x)
    value: Tensor = Wv(x)
    
    # reshape batch, num_heads, seq_len, head_dim 
    key = key.reshape(batch, num_heads, seq_len, head_dim)
    query = query.reshape(batch, num_heads, seq_len, head_dim)
    value = value.reshape(batch, num_heads, seq_len, head_dim)
    
    mask = torch.triu(torch.ones(1,1,seq_len,seq_len) *-torch.inf,diagonal=1) 
    scores = query@key.transpose(-1,-2) / math.sqrt(head_dim)
    scores = mask+scores
    
    attn_mtx = F.softmax(scores,dim=-1)
    out = attn_mtx@value
    
    output = scaled_graph_attention(query, key, value)

    
    assert torch.isclose(output,out,atol=1e-5).all() , "you need to debug buddy"
    
    print("IT WORKS !!!")
    
ITER = 3
for _ in range(ITER):
    test_attn()