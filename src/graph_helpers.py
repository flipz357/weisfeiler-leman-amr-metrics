import logging
from collections import defaultdict
import networkx as nx
import penman
logger = logging.getLogger("penman")
logger.setLevel(30)


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


def nx_digraph_from_triples(triples, label2attribute=False):
    """converts triples to networkx directed graph
    
    Args:
        triples (list): graph triples [(n1, edge1, n2), (n2, edge2, n3)...]
        label2attribute (boolean): if true then relabel nodes with int ids
                                    and keep old label as node attribute
    Returns:
        nx directed graph, root node
    """

    G = nx.MultiDiGraph()

    if label2attribute:
        nodeids = {}
        i = 1
        for tr in triples:
            n1 = tr[0]
            n2 = tr[2]
            if n1 not in nodeids:
                nodeids[n1] = i
                i += 1
            if n2 not in nodeids:
                nodeids[n2] = i
                i += 1

        for node in nodeids:
            G.add_node(nodeids[node], label=node)

        for tr in triples:
            n1 = nodeids[tr[0]]
            n2 = nodeids[tr[2]]
            G.add_edge(n1, n2, label=tr[1])

        return G, None
    
    root = None
    if triples:
        root = triples[0][0]
    for tr in triples:
        G.add_edge(tr[0], tr[2], label=tr[1])
    
    return G, root


def nx_digraph_to_triples(G):
    """convert nx graph to triples"""

    dat = G.edges(data=True)
    triples = []
    for tr in dat:
        triples.append((tr[0], tr[2]['label'], tr[1]))
    return triples


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


def simplify(triples):
    """simplifies triples s.t. variable nodes are replaced with corresponding \
        concepts

    Args: 
        triples (list): list with triples
    
    Returns:
        list with triples
    """
    
    var_concept = get_var_concept_map(triples)

    newtriples = []
    for tr in triples:
        if tr[1] == ':instance':
            continue
        else:
            s = tr[0]
            t = tr[2]
            if s in var_concept:
                s = var_concept[s]
            if t in var_concept:
                t = var_concept[t]
            newtriples.append((s, tr[1], t))
    
    if not newtriples:
        return triples[0], None
    
    return newtriples, None


def simplify_lossless(triples):
    """simplifies triples s.t. variable nodes are replaced with corresponding
        concepts. This simplification is lossless, by handling corefs, etc.
        I.e., the original graph can be reconstructed.

    Args:
        triples (list): list with triples
    
    Returns:
        list with triples
    """
  
    var_concept = get_var_concept_map(triples)

    concept_count = defaultdict(int)
    for v in var_concept.values():
        concept_count[v] += 1
    for k, v in var_concept.items():
        if concept_count[v] > 0:
            var_concept[k] = v + '###cc' + str(concept_count[v])
            concept_count[v] -= 1
    for tr in triples:
        if tr[2] not in var_concept and tr[2] not in concept_count:
            var_concept[tr[2]] = tr[2] + "###cc1" 

    newtriples = []
    for tr in triples:
        if tr[1] == ':instance' and len(triples) == 1:
            newtriples.append((var_concept[tr[0]], ":instance", var_concept[tr[0]]))
        elif tr[1] == ':instance':
            continue
        else:
            s = tr[0]
            t = tr[2]
            if s in var_concept:
                s = var_concept[s]
            if t in var_concept:
                t = var_concept[t]
            newtriples.append((s, tr[1], t))
    return newtriples, {v:k for k, v in var_concept.items()}
