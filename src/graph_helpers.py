import logging
import networkx as nx
import penman
logger = logging.getLogger("penman")
logger.setLevel(30)


def parse_string_amrs(string_penman_amrs, edge_to_node_transform=False):
    """builds nx medi graphs from amr sembank.
    
    Args:
        string_penman_amrs (list): contains AMR graphs, i.e.
                                ["(x / ... :a (...))"], "(..."]

        add_coref_to_labels (bool): if true then add (redundant) 
                                    coref info to node labels (default: False)

    Returns:
        - list with nx multi edge di graph where nodes are ids and nodes and 
            labels carry attribute
        - list with dicts that contain node id --> original AMR variable mappings
    """
    
    amrs = [stringamr2graph(s) for s in string_penman_amrs]
    triples = [graph2triples(G) for G in amrs]
    graphs_nm = [amrtriples2nxmedigraph(tx, edge_to_node_transform) for tx in triples]
    graphs = [elm[0] for elm in graphs_nm]
    node_map = [elm[1] for elm in graphs_nm]

    return graphs, node_map


def amrtriples2nxmedigraph(triples, edge_to_node_transform=False):
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

    if edge_to_node_transform:
        do_edge_node_transform(triples)

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
        inst = triples[cid][2]
        varname = newvarkey + str(idx) + "[==instance:{}]".format(inst)
        triples.append((triples[cid][0], triples[cid][1], varname))
        triples.append((varname, ":instance", inst))
        idx += 1
    for i in reversed(sorted(list(collect_ids))):
        del triples[i]
    return None

def do_edge_node_transform(triples):
    # constant nodes are targets with no outgoing edge (leaves) 
    # that don't have an incoming :instance edge
    collect_ids = set()
    collect_new_triples = []
    newvarkey = "edge_node_"
    for i, tr in enumerate(triples):
        src = tr[0]
        rel = tr[1]
        target = tr[2]
        if rel == ":instance":
            continue
        newnode = newvarkey + str(i) + "[==rel_triple:{}]".format((src, rel, target))
        et1 = (src, ":edge", newnode)
        et2 = (newnode, ":edge", target)
        et3 = (newnode, ":instance", rel)
        collect_ids.add(i)
        collect_new_triples.append(et1)
        collect_new_triples.append(et2)
        collect_new_triples.append(et3)
    triples += collect_new_triples
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


