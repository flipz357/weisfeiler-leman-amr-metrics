import numpy as np
import logging
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import entropy
from pyemd import emd_with_flow
import gensim.downloader as api
from copy import deepcopy
import graph_helpers 
import multiprocessing
import re

logger = logging.getLogger(__name__)


class GraphSimilarityPredictor():

    def predict(self, amrs1, amrs2):
        """predicts similarities for paired amrs
        Input are multi edge networkx Di-Graphs 
        """
        pass


class Preprocessor():

    def transform(self, triples_1, triples_2):
        """predicts similarities for paired amrs"""
        pass


class PreparablePreprocessor(Preprocessor):

    def prepare(self, triples_1, triples_2):
        """ prepare something on triple data, e.g. load embeds"""
        pass
 

class AmrWasserPreProcessor(PreparablePreprocessor):

    def __init__(self, w2v_uri='glove-wiki-gigaword-100', 
                    relation_type="scalar", init="random_uniform"):
        """Initilize Preprocessor object

        Args:
            w2v_uri (string): uri to desired word embedding
                              e.g., 
                              'word2vec-google-news-100'
                              'glove-twitter-100'
                              'fasttext-wiki-news-subwords-300'
                              etc.
                              if None, then use only random embeddings
            relation_type (string): edge label representation type
                                    possible: either 'scalar' or 'vector'
        """

        w2v_uri = w2v_uri
        self.dim = 100
        self.wordvecs = {}
        self.relation_type = relation_type
        self.params_ready = False
        self.init = init
        self.param_keys = None
        self.params = None
        self.unk_nodes = None
        if w2v_uri:
            try:
                self.wordvecs = api.load(w2v_uri)
                self.dim = self.wordvecs["dog"].shape[0]
            except ValueError:
                logger.warning("tried to load word embeddings specified as '{}'.\
                        these are not available. all embeddings will be random.\
                        If this is desired, no need to worry.".format(w2v_uri))
        return None

    def prepare(self, triples_1, triples_2):
        """initialize embedding of graph nodes and edges"""
        self.params_ready = False
        _, _, _, _ = self.transform_return_node_map(triples_1, triples_2)
        self.params_ready = True
        return None

    def transform(self, triples_1, triples_2):
        """transforms triple graphs to nx multi-edge-di graphs
        
        Args:
            triples_1 (list): list with graph as triples
            triples_2 (list): list with graph as triples

        Returns:
            nx multi-edge di graphs with embedded nodes
        """

        amrs1, amrs2, _, _ = self.transform_return_node_map(triples_1, triples_2)
        return amrs1, amrs2

    def transform_return_node_map(self, triples_1, triples_2):
        """transforms triple graphs to nx multi-digraphs AND returns a map 
            from graph nodes to original variable nodes of the AMR so 
            that an alignment can be reconstructed

        Args:
            triples_1 (list): list with graph as triples
            triples_2 (list): list with graph as triples

        Returns:
            nx multi-edge di graphs with embedded nodes, node map from
            graph nodes to original AMR variable nodes
        """
        
        def _simp_create_nx(triples):
            # lossless simplification of graphs, and get maaping to original graph
            triples_node2nodeorig = [graph_helpers.simplify_lossless(G) for G in triples]
            triples = [x[0] for x in triples_node2nodeorig]
            node2nodeorig = [x[1] for x in triples_node2nodeorig]
            # build nx multi edge di graphs
            amrs = [graph_helpers.nx_digraph_from_triples(G, label2attribute=True)[0] for G in triples]
            return amrs, node2nodeorig
        
        amrs_1, node2nodeorig_1 = _simp_create_nx(triples_1)
        amrs_2, node2nodeorig_2 = _simp_create_nx(triples_2)
        
        #gather embeddings for nodes and edges
        self.embeds(amrs_1, amrs_2)

        return amrs_1, amrs_2, node2nodeorig_1, node2nodeorig_2
     
    def embeds(self, gs1, gs2):
        """ embeds all graphs, i.e., assign embeddings to node labels 
            and edge labels

        Args:
            gs1 (list with nx medi graphs): a list of graphs
            gs2 (list with nx medi graphs): a list of graphs

        Returns:
            None
        """

        if not self.params_ready:
            self.param_keys = {}
            self.params = []
            self.unk_nodes = {}
        for g in gs1:
            self.embed(g)
        for g in gs2:
            self.embed(g)
        self.params = np.array(self.params)
        return None
     
    def get_vec_deprecated(self, string):
        """DEPRECATED""" 
         
        string  = string.split("-")[0]

        # if the node is a negation node in AMR (indicated as '-', we assign
        # the word vector for 'no')
        if not string:
            string = "no"

        # further cleaning
        string = string.split("###")[0].strip()
        string = string.replace("\"", "").replace("'", "")
        string = string.lower()

        #lookup
        if string in self.wordvecs:
            return np.copy(self.wordvecs[string])
        return None
    
    def get_vec(self, string):
        """lookup a vector for a string
        
        Args:
            string (str): a string
        
        Returns: 
            n-dimensional numpy vector
        """ 
        
        string = string.strip()

        # if the node is a negation node in AMR (indicated as '-', we assign
        # the word vector for 'no')
        if string == "-":
            string = "no"

        # further cleaning
        string = string.split("###")[0].strip()
        string = string.replace("\"", "").replace("'", "")
        string = string.lower()

        # we can delete word senses here (since they will get contextualized)
        string = re.sub(r'-[0-9]+', '', string)
        vecs = [] 
        
        #lookup
        for key in string.split("-"):
            if key in self.wordvecs:
                vecs.append(np.copy(self.wordvecs[key]))
        if vecs:
            return np.max(vecs, axis=0)

        return None
    
    def embed(self, G):
        """in-place manipulation of nx medi graph 
        by assigning embeddings to edges and nodes. 
        This fucntion also initializes/updates/gathers 'global' edge parameters
        
        Args:
            G (nx medi graph): an nx multi edge directed graph

        Returns:
            None
        """

        label_vec = {}
        label_no_vec = set()
        def _maybe_link_labels_vecs(graph):
            for node in graph:
                label = graph.nodes[node]["label"]
                vec = self.get_vec(label)
                if vec is not None:
                    label_vec[label] = vec
                else:
                    label_no_vec.add(label)
            for (n1, n2, _) in graph.edges:
                edat = graph.get_edge_data(n1, n2)
                for k in edat:
                    label = edat[k]["label"]
                    label_no_vec.add(label)
            return None
        
        _maybe_link_labels_vecs(G)
        
        def _handle_unks():
        
            for la in label_no_vec:
                if la[0] == ":":    
                    if la not in self.param_keys:
                        if not self.params_ready:
                            if self.relation_type == "scalar":
                                if self.init == "random_uniform":
                                    self.params.append(np.random.uniform(0.0, 1, size=(1)))
                                
                                elif self.init == "min_entropy": 
                                    sample = []
                                    for _ in range(10):
                                        sample.append(np.random.uniform(0.0, 1, size=(1)))
                                    entropies = []
                                    for i in range(10):
                                        entropies.append(
                                                entropy(np.array(np.array(self.params).flatten() 
                                                            + list(sample[i])).flatten()))
                                    argmin = np.argmin(entropies)
                                    self.params.append(sample[argmin])
                                
                            if self.relation_type == "vector":
                                self.params.append(np.random.rand(self.dim))
                            if self.relation_type == "matrix":
                                self.params.append(np.random.rand(self.dim, self.dim))
                            self.param_keys[la] = len(self.params) - 1
                else:
                    if la not in self.unk_nodes:
                        if not self.params_ready:
                            self.unk_nodes[la] = np.random.rand(self.dim)
                    if la in self.unk_nodes:
                        label_vec[la] = self.unk_nodes[la]
                    else:
                        label_vec[la] = np.random.rand(self.dim)
            return None

        _handle_unks()
        
        def _make_latent(graph):
            for node in graph.nodes:
                label = graph.nodes[node]["label"]
                graph.nodes[node]["latent"] = label_vec[label]
            return None

        _make_latent(G)
        
        return None

class AmrWasserPredictor(GraphSimilarityPredictor):

    def __init__(self, params=None, param_keys=None, iters=2):
        self.params = params
        if params is None:
            self.params = []
        self.param_keys = param_keys
        if param_keys is None:
            self.param_keys = {}
        self.unk_edge = np.random.rand(self.params.shape[1]) 
        self.iters = iters
        return None
  
    def _predict_single(self, graphtuple):
        """Predict WWLK similarity for a (A,B) graph tuple
        
        Args:
            graphtuple (tuple): graph tuple (A,B) of nx medi graphs
            iters (int): what degree of contextualization is desired?
                        default=2

        Returns:
            similarity
        """

        a1 = graphtuple[0]
        a2 = graphtuple[1]
        
        #get init node embeddings
        e1, _ = self.collect_graph_embed(a1)
        e2, _ = self.collect_graph_embed(a2)
    
        #get node embeddings from different k
        E1, E2 = self.WL_latent(a1, a2, iters=self.iters)

        #concat
        E1 = np.concatenate([e1, E1], axis=1)
        E2 = np.concatenate([e2, E2], axis=1)
       
        # get wmd input
        v1, v2, dists = self.get_wmd_input(E1, E2)

        #compute wmd
        emd, _ = emd_with_flow(v1, v2, dists)
        
        return emd * -1
    
    def _predict_single_flow_order(self, graphtuple):
        """Predict WWLK similarity for a (A,B) graph tuple and return alignment
        
        Args:
            graphtuple (tuple): graph tuple (A,B) of nx medi graphs
            iters (int): what degree of contextualization is desired?
                        default=2

        Returns:
            - similarity
            - flow matrix
            - dist matrix
            - indeces of matrices
        """

        a1 = graphtuple[0]
        a2 = graphtuple[1]

        e1, order1 = self.collect_graph_embed(a1)
        e2, order2 = self.collect_graph_embed(a2)
         
        E1, E2 = self.WL_latent(a1, a2, iters=2)
        E1 = np.concatenate([e1, E1], axis=1)
        E2 = np.concatenate([e2, E2], axis=1)
       
        v1, v2, dists = self.get_wmd_input(E1, E2)
        emd, flow = emd_with_flow(v1, v2, dists)
        
        return emd * -1, flow, dists, order1, order2

    def predict(self, amrs1, amrs2, parallel=False):
        
        """Predict WWLK similarities for two (parallel) data sets
        
        Args:
            amrs1 (list): list with nx medi graphs a_1,...,a_n
            amrs2 (list): list with nx medi graphs b_1,...,b_n
            parallel (boolean): parallelize computation? default=no

        Returns:
            similarities for AMR graphs
        """
        
        assert len(amrs2) == len(amrs1)
        
        amrs1 = deepcopy(amrs1)
        amrs2 = deepcopy(amrs2)
        
        zipped = list(zip(amrs1, amrs2))
        preds = []

        if parallel:
            with multiprocessing.Pool(10) as p:
                preds = p.map(self._predict_single, zipped)
        else:
            for i in range(len(zipped)):    
                preds.append(self._predict_single(zipped[i]))

        return np.array(preds)

    def predict_and_align(self, amrs1, amrs2, nodemap1, nodemap2):
        """Predict WWLK similarities for two (parallel) data sets 
            and get alignments
        
        Args:
            amrs1 (list): list with nx medi graphs a_1,...,a_n
            amrs2 (list): list with nx medi graphs b_1,...,b_n
            nodemap1 (dict): mapping from nx medi graph nodes 
                             to standard AMR variables for amrs1
            nodemap2 (dict): mapping from nx medi graph nodes 
                             to standard AMR variables for amrs2

        Returns:
            - similarities for AMR graphs
            - alignments
        """

        node2nodeorig_1 = nodemap1
        node2nodeorig_2 = nodemap2
        amrs1 = deepcopy(amrs1)
        amrs2 = deepcopy(amrs2)
        zipped = list(zip(amrs1, amrs2))
        assert len(amrs2) == len(amrs1)
        
        preds = []
        aligns = []
        for i in range(len(zipped)):
            #get sims, flows, etc
            sim, flow, dists, order1, order2 = self._predict_single_flow_order(zipped[i])
            align_dict = {} 
            # project alignment to orig AMR graphs
            for j, label in enumerate(order1):
                align_dict[node2nodeorig_1[i][label]] = {}
                row = flow[j][len(order1):]
                cost_row = dists[j][len(order1):]
                for k, num in enumerate(row):
                    if num > 0.0:
                        align_dict[node2nodeorig_1[i][label]][node2nodeorig_2[i][order2[k]]] = (num, cost_row[k])
            aligns.append(align_dict)
            preds.append(sim)

        return preds, aligns

    def get_wmd_input(self, mat1, mat2):
        """Prepares input for pyemd
        
        Args:
            mat1 (matrix): embeddings fror nodes x_1,...,x_n
            mat2 (matrix): embeddings fror nodes y_1,...,y_m

        Returns:
            - prior weights for nodes of A
            - prior weights for nodes of B
            - cost matrix
        """
        
        # construct prior weights of nodes... all are set equal here
        v1 = np.concatenate([np.ones(mat1.shape[0]), np.zeros(mat2.shape[0])])
        v2 = np.concatenate([np.zeros(mat1.shape[0]), np.ones(mat2.shape[0])])
        v1 = v1 / sum(v1)
        v2 = v2 / sum(v2)
        
        # build cost matrix
        dist = np.zeros(shape=(len(v1), len(v1)))
        vocab_map = {}
        for i in range(v1.shape[0]):
            if i < mat1.shape[0]:
                vocab_map[i] = mat1[i]
            if i >= mat1.shape[0]:
                vocab_map[i] = mat2[i - mat1.shape[0]]
        for i in range(dist.shape[0]):
            for j in range(dist.shape[1]):
                dist[i, j] = euclidean(self.norm(vocab_map[i]), self.norm(vocab_map[j]))
        return v1, v2, dist

    
    def norm(self, x):
        """scale vector to length 1"""
        div = np.linalg.norm(x)
        return x / div

    def set_params(self, params, idx=None):
        """set edge params"""
        if idx is None:
            self.params = params
        else:
            self.params[idx] = params
        return None
    
    def get_params(self):
        """get edge params"""
        return self.params
     
    def collect_graph_embed(self, nx_latent):
        """collect the node embeddings from a graph
        
        Args:
            nx_latent (nx medi graph): a graph that has node embeddings 
                                        as attributes
        Returns:
            - the node embeddings
            - labels of nodes
        """

        vecs = []
        labels = []
        for node in nx_latent.nodes:
            vecs.append(nx_latent.nodes[node]["latent"])
            labels.append(nx_latent.nodes[node]["label"])
        return np.array(vecs), labels

    def maybe_has_param(self, label):
        """safe retrieval of an edge parameter"""
        if label not in self.param_keys:
            return self.unk_edge
        else:
            return self.params[self.param_keys[label]]

    def _communicate(self, G, node):
        """In-place contextualization of a node with neighborhood

        Args:
            G (nx medi graph): input graph
            node: the node

        Returns:
            None
        """

        summ = []
        
        def mult_reduce(mat):
            newvec = mat[0].copy()
            for v in mat[1:]:
                newvec *= v
            return newvec
        # iterate over outgoing
        for nb in G.neighbors(node):
            # get node embedding of neighbor
            latents = [G.nodes[nb]["latent"]]
            # multi di graph can have multiple edges, so iterate over all
            for k in G.get_edge_data(node, nb):
                e_latent = self.maybe_has_param(G.get_edge_data(node, nb)[k]['label'])
                latents.append(e_latent)
            #scalar or vector multiplication with edge params and neighbor node embedding
            #latents = np.multiply.reduce(latents, axis=0, dtype=object)
            latents = mult_reduce(latents)
            summ.append(latents)
              
        # iterate over incoming, see above
        for nb in G.predecessors(node):
            latents = [G.nodes[nb]["latent"]]
            for k in G.get_edge_data(nb, node):
                e_latent = self.maybe_has_param(G.get_edge_data(nb, node)[k]['label'])
                latents.append(e_latent)
            #latents = np.multiply.reduce(latents, axis=0, dtype=object)
            latents = mult_reduce(latents)
            summ.append(latents)
        
        # handle possible exception case
        if not summ:
            summ = np.zeros((1, G.nodes[node]["latent"].shape[0]))

        # compute new embedding of node
        summ = np.mean(summ, axis=0)
        G.nodes[node]["newlatent"] = G.nodes[node]["latent"] + summ
        return None

    def communicate(self, nx_latent):
        """Applies contextualization (in-place) for all nodes of a graph
        
        Args:
            G (nx medi graph): input graph

        Returns:
            None
        """
        
        #collect new embeddings
        for node in nx_latent.nodes:
            self._communicate(nx_latent, node)

        #set new embeddings
        for node in nx_latent.nodes:
            nx_latent.nodes[node]["latent"] = nx_latent.nodes[node]["newlatent"]
        
        return None

    def wl_iter_latent(self, nx_g1_latent, nx_g2_latent):
        """apply one WL iteration and get node embeddings for two graphs A and B

        Args:
            nx_g1_latent (nx medi graph): graph A
            nx_g2_latent (nx medi graph): graph B

        Returns:
            - contextualized node embeddings for A
            - contextualized node embeddings for B
            - new copy of A
            - new copy of B
        """

        g1_copy = deepcopy(nx_g1_latent)
        g2_copy = deepcopy(nx_g2_latent)
        self.communicate(g1_copy)
        self.communicate(g2_copy)
        mat1, _ = self.collect_graph_embed(g1_copy)
        mat2, _ = self.collect_graph_embed(g2_copy)
        return mat1, mat2, g1_copy, g2_copy

    def WL_latent(self, nx_g1_latent, nx_g2_latent, iters=2):
        """apply K WL iteration and get node embeddings for two graphs A and B

        Args:
            nx_g1_latent (nx medi graph): graph A
            nx_g2_latent (nx medi graph): graph B

        Returns:
            - contextualized node embeddings for A for k=1,..k=n
            - contextualized node embeddings for B for k=1,..k=n
        """

        v1s = []
        v2s = []
        for _ in range(iters):
            x1_mat, x2_mat, nx_g1_latent, nx_g2_latent = self.wl_iter_latent(nx_g1_latent, nx_g2_latent)
            v1s.append(x1_mat)
            v2s.append(x2_mat)
        g_embed1 = np.concatenate(v1s, axis=1)
        g_embed2 = np.concatenate(v2s, axis=1)
        return g_embed1, g_embed2


class AmrSymbolicPreprocessor(Preprocessor):

    def transform(self, triples_1, triples_2):
        """ builds multi edge nx digraphs from graphs as triples
        
        Args: 
            triples1 (list): list with graphs
            triples2 (list): list with graphs

        Returns:
            two lists with multi edge nx digraphs
        """
        
        amrs1 = [graph_helpers.simplify_lossless(G)[0] for G in triples_1]
        amrs1 = [graph_helpers.nx_digraph_from_triples(G)[0] for G in amrs1]
        amrs2 = [graph_helpers.simplify_lossless(G)[0] for G in triples_2]
        amrs2 = [graph_helpers.nx_digraph_from_triples(G)[0] for G in amrs2]
        assert len(amrs2) == len(amrs1)
        
        return amrs1, amrs2


class AmrSymbolicPredictor(GraphSimilarityPredictor):
    
    def __init__(self, simfun='cosine', iters=2):
        self.simfun = simfun
        self.iters = iters

    def predict(self, amrs1, amrs2):
        """predicts similarity scores for paired graphs
        
        Args:
            amrs1 (list with nx medi graphs): graphs
            amrs2 (list with nx medi graphs): other graphs

        Returns:
            list with floats (similarities)
        """

        kvs = []
        for i, a1 in enumerate(amrs1):
            a2 = amrs2[i]
            gs1 = self.get_stats(a1, a2, stattype='nodecount')
            gs2 = self.get_stats(a1, a2, stattype='triplecount')
            v1, v2 = gs1[0], gs1[1]
            v1, v2 = np.concatenate([v1, gs2[0]]), np.concatenate([v2, gs2[1]])
            kv = self.wlk(a1, a2, iters=self.iters, kt=self.simfun, init_vecs=(v1, v2), 
                            weighting="linear", stattype="nodecount")
            if np.isnan(kv):
                kv = 0.0
            kvs.append(kv)
        return kvs
        
    def wl_gather_node(self, node, G):
        """ gather edges+labels for a node from the neighborhood
        
        Args:
            node (hashable object): a node of the graph
            G (nx medi graph): the graph

        Returns:
            a list with edge+label from neighbors
        """
        
        newn = [node]

        for nb in G.neighbors(node):
            for k in G.get_edge_data(node, nb):
                el = G.get_edge_data(node, nb)[k]['label']
                newn.append(el + '_' + nb)
        
        for nb in G.predecessors(node):
            for k in G.get_edge_data(nb, node):
                el = G.get_edge_data(nb, node)[k]['label']
                newn.append(el + '_' + nb)
        
        return newn
    
    def wl_gather_nodes(self, G):
        """apply gathering (wl_gather_node) for all nodes
        
        Args:
            G (nx medi graph): the graph

        Returns:
            a dictionary node -> neigjborhood
        """

        dic = {}
        for n in G.nodes:
            dic[n] = self.wl_gather_node(n, G)
        return dic
    
    def sort_relabel(self, dic1, dic2):
        """form aggregate labels via sorting
        
        Args: 
            dic1 (dict): node-neighborhood dict of graph A
            dic2 (dict): node-neighborhood dict of graph B
            
        Returns:
            two dicts where keys are same and values are strings
        """

        for node in dic1:
            dic1[node] = ' ::: '.join(list(sorted(dic1[node])))

        for node in dic2:
            dic2[node] = ' ::: '.join(list(sorted(dic2[node])))

        return dic1, dic2
    
    def get_triples_with_new_nodes(self, nxg, nn_dict):
        """construct new graph triples with new nodes

        Args:
            nxg (nx medi graph): graph
            nn_dict (dict): node->node dict

        Returns:
            triples with new nodes
        """

        newtr = []
        triples = graph_helpers.nx_digraph_to_triples(nxg)
        for s, e, t in triples:
            sbar = nn_dict[s]
            tbar = nn_dict[t]
            newtr.append((sbar, e, tbar))
        return newtr
 
    def wlk(self, nx_g1, nx_g2, iters=2, weighting='linear', kt='dot', 
            stattype='nodecount', init_vecs=(None, None)):
        """compute WL kernel similarity of graph A and B

        Args:
            nx_g1 (nx medi graph): graph A
            nx_g2 (nx medi graph): graph B
            iters (int): iterations
            weighting (string): decrease weight of iteration stats
            kt (string): kernel type, default dot
            stattype (string): which features? default: nodecount
            init_vecs (tuple): perhaps there are already 
                             some features for A and B?
        
        Returns:
            kernel similarity
        """

        v1s, v2s, _ = self.wl(nx_g1, nx_g2, iters=iters, stattype=stattype)
        if init_vecs[0] is not None:
            v1s = [
             init_vecs[0]] + v1s
            v2s = [init_vecs[1]] + v2s
        if weighting == 'exp':
            wts = np.array([np.e ** (-1 * x) for x in range(0, 100)])
            wts = wts[:len(v1s)]
            v1s = [vec * wts[i] for i, vec in enumerate(v1s)]
            v2s = [vec * wts[i] for i, vec in enumerate(v2s)]
        if weighting == 'linear':
            wts = np.array([1 / (1 + x) for x in range(0, 100)])
            wts = wts[:len(v1s)]
            v1s = [vec * wts[i] for i, vec in enumerate(v1s)]
            v2s = [vec * wts[i] for i, vec in enumerate(v2s)]
        v1 = np.concatenate(v1s)
        v2 = np.concatenate(v2s)
        if kt == 'dot':
            return np.einsum('i,i ->', v1, v2)
        if kt == 'cosine':
            return 1 - cosine(v1, v2)
        if kt == 'rbf':
            gamma = 2.5
            diff = v1 - v2
            dot = -1 * np.einsum('i,i ->', diff, diff)
            div = 2 * gamma ** 2
            return np.exp(dot / div)

    def wl(self, nx_g1, nx_g2, iters=2, stattype='nodecount'):
        """collect vectors over WL iterations

        Args:
            nx_g1 (nx medi graph): graph A
            nx_g2 (nx medi graph): graph B

        Returns:
            a list for every graph that contains vectors
        """

        v1s = []
        v2s = []
        vocabs = []
        for _ in range(iters):
            x1, x2, nx_g1, nx_g2, vocab = self.wl_iter(nx_g1, nx_g2, stattype=stattype)
            v1s.append(x1)
            v2s.append(x2)
            vocabs.append(vocab)
        return v1s, v2s, vocabs

    def wl_iter(self, nx_g1, nx_g2, stattype='nodecount'):
        """collect vectors over one WL iteration

        Args:
            nx_g1 (nx medi graph): graph A
            nx_g2 (nx medi graph): graph B

        Returns:
            - a list for every graph that contains vectors
            - new aggreagate graphs
        """

        dic_g1 = self.wl_gather_nodes(nx_g1)
        dic_g2 = self.wl_gather_nodes(nx_g2)
        d1, d2 = self.sort_relabel(dic_g1, dic_g2)
        newtriples1 = self.get_triples_with_new_nodes(nx_g1, d1)
        newtriples2 = self.get_triples_with_new_nodes(nx_g2, d2)
        newg1, _ = graph_helpers.nx_digraph_from_triples(newtriples1)
        newg2, _ = graph_helpers.nx_digraph_from_triples(newtriples2)
        stats1, stats2, vocab = self.get_stats(newg1, newg2, stattype=stattype)
        return stats1, stats2, newg1, newg2, vocab
    
    def get_stats(self, g1, g2, stattype='nodecount'):
        """get feature vec for a statistitic type
        
        Args: 
            g1 (nx medi graph): graph A
            g2 (nx medi graph): graph B
            stattype (string): statistics type, default: node count

        Returns:
            - vector for A
            - vector for B
            - vocab
        """

        if stattype == 'nodecount':
            return self.nc(g1, g2)
        if stattype == 'triplecount':
            return self.tc(g1, g2)
    
    def create_fea_vec(self, items, vocab):
        """create freture vector from bow list and vocab
        
        Args:
            items (list): list with items e.g. [x, y, z]
            vocab (dict): dict with item-> id eg. {x:2, y:4, z:5}

        Returns:
            feature vector, e.g., [0, 0, 1, 0, 1, 1]
        """

        vec = np.zeros(len(vocab))
        for item in items:
            vec[vocab[item]] += 1
        return vec

    def nc(self, g1, g2):
        """ feature vector constructor for node BOW of two graphs
        
        Args:
            g1 (nx medi graph): graph A
            g2 (nx medi graph): graph B
        
        Returns:
            feature vector for graph A, feature vector for graph B, vocab 
        """
        
        vocab = {}
        i = 0
        g1bow = []
        g2bow = []
        for node in g1.nodes:
            g1bow.append(node)
            if node not in vocab:
                vocab[node] = i
                i += 1
        for node in g2.nodes:
            g2bow.append(node)
            if node not in vocab:
                vocab[node] = i
                i += 1

        vec1 = self.create_fea_vec(g1bow, vocab)
        vec2 = self.create_fea_vec(g2bow, vocab)
        return (vec1, vec2, vocab)

    def tc(self, g1, g2):
        """ feature vector constructor for triple BOW of two graphs
        
        Args:
            g1 (nx medi graph): graph A
            g2 (nx medi graph): graph B
        
        Returns:
            feature vector for graph A, feature vector for graph B, vocab 
        """

        vocab = {}
        g1bow = []
        g2bow = []
        i = 0
        for node in g1.nodes:
            for nb in g1.neighbors(node):
                for k in g1.get_edge_data(node, nb):
                    el = node
                    el += '_' + g1.get_edge_data(node, nb)[k]['label'] 
                    el += '_' + nb
                    g1bow.append(el)
                    if el not in vocab:
                        vocab[el] = i
                        i += 1
        for node in g2.nodes:
            for nb in g2.neighbors(node):
                for k in g2.get_edge_data(node, nb):
                    el = node 
                    el += '_' + g2.get_edge_data(node, nb)[k]['label'] 
                    el += '_' + nb
                    g2bow.append(el)
                    if el not in vocab:
                        vocab[el] = i
                        i += 1
 
        vec1 = self.create_fea_vec(g1bow, vocab)
        vec2 = self.create_fea_vec(g2bow, vocab)

        return vec1, vec2, vocab
