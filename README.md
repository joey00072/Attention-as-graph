# Attention-as-graph
alternative way to calculating self attention


> [!WARNING]
> I may or may not work on it further, PR are welcome though


look `main.py` this is preview

```python

@dataclass
class Node:
    idx: int
    value: Tensor
    adjacency_list: list[Edge] = field(default_factory=list)

@dataclass
class Edge:
    node: Node
    weight: Tensor



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
   ```


## TODO
- [ ] so inferace with tiny lm as poc
- [ ] Add visuization
    - top nodes infuanceing current node



---
its for education purpose, has no pratical use (unless added visualiztion)