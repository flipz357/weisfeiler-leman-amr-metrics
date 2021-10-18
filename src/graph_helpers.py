import logging
from collections import defaultdict
import networkx as nx
import penman
logger = logging.getLogger("penman")
logger.setLevel(30)


def amrtriples2nxmedigraph(triples, add_coref_info_to_labels=False):
    """builds nx medi graph from amr triples.
    
    Args:
        triples (list): list with AMR triples, 
                        e.g. [("a", ":instance", "boy"), ("r", ":arg0", "b"), ...]
        add_coref_to_labels (bool): if true then add (redundant) 
                                    coref info to node labels (default: False)

    Returns:
        nx multi edge di graph where nodes are ids and nodes and labels carry attribute
        "label".
    """

    # reify nodes (e.g., (n, :op1, US) ---> (n, :op1, var) & (var, :instance, US))
    reify_nodes(triples)

    # build variable -> concept map
    var_concept_map = get_var_concept_map(triples)
    
    # build variable -> index map
    var_index_map = get_var_index_map(triples)
    
    # build index -> variable map
    index_var_map = {v:k for k, v in var_index_map.items()}
    
    # build index -> concept map
    index_concept_map = {k:var_concept_map[index_var_map[k]] for k in index_var_map}
    
    #init graph
    G = nx.MultiDiGraph()
    
    # add nodes
    add_nodes(G, index_var_map.keys(), index_concept_map)
    
    # add edges
    add_edges(G, [t for t in triples if t[1] != ":instance"], var_index_map)
    
    # maybe add coref info to node atribute "label"
    if add_coref_info_to_labels:
        G = add_coref_label(G)

    # return graph and a map from node ids to orig. AMR variables
    return G, index_var_map


def add_nodes(G, nodelist, label_map):
    """ add nodes to a graph

    Args:
        G (nx medi graph): input graph
        nodelist (list): a list with node ids to be inserted into G
        label_map (dict): a map node id --> label (e.g., {0:"boy", ...})

    Returns:
        None
    """

    for n in nodelist:
        G.add_node(n, label=label_map[n])
    return None


def add_edges(G, triples, src_tgt_index_map):
    """ add edges to graph.

    Args:
        G (nx medi graph): a graph
        triples (list): list with (s, rel, t) tuples
        src_tgt_index_map (dict): a map from amr variables to node ids

    Returns:
        None
    """

    for tr in triples:
        src = src_tgt_index_map[tr[0]]
        label = tr[1]
        try:
            tgt = src_tgt_index_map[tr[2]]
        except KeyError:
            # handle very rare case where a constant 
            # also appears with an incoming isinstance edge
            found = 0
            for n in G.nodes:
                if G.nodes[n]["label"] == tr[2]:
                    found = n
            if found:
                tgt = found
            else:
                continue
        G.add_edge(src, tgt, label=label)
    return None


def reify_nodes(triples):
    # constant nodes are targets with no outgoing edge (leaves) 
    # that don't have an incoming :instance edge
    collect_ids = set()
    for i, tr in enumerate(triples):
        target = tr[2]
        incoming_instance = False
        for tr2 in triples:
            if tr2[1] == ":instance" and tr2[0] == target:
                incoming_instance = True
            if tr2[1] == ":instance" and tr2[2] == target:
                incoming_instance = True
        if not incoming_instance:
            collect_ids.add(i)
    newvarkey = "rfattribute_"
    idx = 0
    for cid in collect_ids:
        varname = newvarkey + str(idx)
        triples.append((triples[cid][0], triples[cid][1], varname))
        triples.append((varname, ":instance", triples[cid][2]))
        idx += 1
    for i in reversed(sorted(list(collect_ids))):
        del triples[i]
    return None


def stringamr2graph(string):
    """uses penman to convert serialized AMR to penman graph
    
    Args:
        string (str): serialized AMR '(n / concept :arg1 ()...)'
    
    Returns:
        penman graph object
    """
    
    #decode
    g = penman.decode(string)
    
    #handle potential cases where vaiable names in AMR are also concept names
    for i, tr in enumerate(g.triples):
        if tr[1] == ":instance" and tr[0] == tr[2]:
            g.triples[i] = (tr[0], tr[1], tr[2] + "_")
    
    return g


def triples2graph(triples):
    return penman.Graph(triples)


def graph2triples(G):
    tr = G.triples
    return tr


def get_var_concept_map(triples):
    """creates a dictionary that maps varibales to their concepts
        e.g., [(x, :instance, y),...] ---> {x:y,...}

    Args:
        triples (list): triples

    Returns:
        dictionary
    """
    
    var_concept = {}
    for tr in triples:
        if tr[1] == ':instance':
            var_concept[tr[0]] = tr[2]
    return var_concept


def get_var_index_map(triples):
    """creates a dictionary that maps varibales to indeces
        e.g., [(x, :instance, y),...] ---> {x:y,...}

    Args:
        triples (list): triples

    Returns:
        dictionary
    """
    
    var_index = {}
    idx = 0
    for tr in triples:
        if tr[1] == ':instance':
            var_index[tr[0]] = idx
            idx += 1
    return var_index


def nx_digraph_to_triples(G):
    """convert nx graph to triples. Attention: there may be info loss"""

    dat = G.edges(data=True)
    triples = []
    for tr in dat:
        #print(tr)
        src_label = G.nodes[tr[0]]["label"]
        tgt_label = G.nodes[tr[1]]["label"]
        triples.append((src_label, tr[2]['label'], tgt_label))
    return triples


def add_coref_label(G):
    """add coref info to node labels.

    E.g., node 1 with label "apple" and node 2 with label "apple"
    ----> node 1 has label "apple###2" and node 2 has label "apple###1"
    
    Args:
        G (nx medi graph): nx multi edge di graph that has "label" attribute on
                            every node

    Returns:
        a new graph
    """

    G = G.copy()
    labelcount = defaultdict(int)
    for node in G.nodes:
        label = G.nodes[node]["label"]
        labelcount[label] += 1
    for node in G.nodes:
        label = G.nodes[node]["label"]
        G.nodes[node]["label"] = label + "###" + str(labelcount[label])
        labelcount[label] -= 1 
    node_idx = {}
    triples = nx_digraph_to_triples(G)
    idx = 0
    for tr in triples:
        if tr[0] not in node_idx:
            node_idx[tr[0]] = idx
            idx += 1
        if tr[2] not in node_idx:
            node_idx[tr[2]] = idx
            idx += 1
    G_new = nx.MultiDiGraph()
    for node in node_idx:
        G_new.add_node(node_idx[node], label=node)
    add_edges(G_new, triples, node_idx)
    if not triples:
        for node in G:
            G_new.add_node(node, label=G.nodes[node]["label"])
    return G_new

