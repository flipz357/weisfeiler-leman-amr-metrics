import numpy as np
import logging
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import entropy
from pyemd import emd_with_flow
import gensim.downloader as api
from copy import deepcopy
import multiprocessing
import re
from collections import Counter

logger = logging.getLogger(__name__)


class GraphSimilarityPredictor():

    def predict(self, graphs_1, graphs_2):
        """predicts similarities for paired amrs
        Input are multi edge networkx Di-Graphs 
        """
        pass


class Preprocessor():

    def transform(self, graphs_1, graphs_2):
        """preprocesses amr graphs"""
        pass


class PreparablePreprocessor(Preprocessor):

    def prepare(self, graphs_1, graphs_2):
        """ prepare something on graph data, e.g. load embeds"""
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
        self.unk_nodes = {}
        if w2v_uri:
            try:
                self.wordvecs = api.load(w2v_uri)
                self.dim = self.wordvecs["dog"].shape[0]
            except ValueError:
                logger.warning("tried to load word embeddings specified as '{}'.\
                        these are not available. all embeddings will be random.\
                        If this is desired, no need to worry.".format(w2v_uri))
        return None

    def prepare(self, graphs_1, graphs_2):
        """initialize embedding of graph nodes and edges"""
        self.params_ready = False
        self._prepare(graphs_1, graphs_2)
        self.params_ready = True
        return None


    def transform(self, graphs1, graphs2):
        """embeds nx multi-digraphs. I.e. assigns every 
        node and edge attribute "latent" with a parameter

        Args:
            graphs1 (list): list withnx medi graph
            graphs2 (list): list with nx medi graph 

        Returns:
            nx multi-edge di graphs with embedded nodes, node map from
            graph nodes to original AMR variable nodes
        """
          
        #gather embeddings for nodes and edges
        self.embeds(graphs1, graphs2)

        return None
     
    def embeds(self, gs1, gs2):
        """ embeds all graphs, i.e., assign embeddings to node labels 
            and edge labels

        Args:
            gs1 (list with nx medi graphs): a list of graphs
            gs2 (list with nx medi graphs): a list of graphs

        Returns:
            None
        """

        for g in gs1:
            self.embed(g)
        for g in gs2:
            self.embed(g)
        return None
    
    def embed(self, G):

        #get unknown nodes 
        label_no_vec = set()
        label_vec = {}
        for node in G:
            label = G.nodes[node]["label"]
            if label in self.unk_nodes:
                label_vec[label] = self.unk_nodes[label]
                continue
            vec = self.get_vec(label)
            if vec is None:
                label_no_vec.add(label)
            else:
                label_vec[label] = vec
        
        #get vecs for new unknown nodes and update
        rand_vecs = np.random.rand(len(label_no_vec), self.dim)
        for i, label in enumerate(label_no_vec):
            vec = rand_vecs[i]
            self.unk_nodes[label] = vec
            label_vec[label] = vec
        
        #set node latent
        def _make_latent(graph):
            for node in graph.nodes:
                label = graph.nodes[node]["label"]
                graph.nodes[node]["latent"] = label_vec[label]
            return None

        _make_latent(G)

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
        # random vec)
        if string == "-":
            return None

        # further cleaning
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

    def _prepare(self, graphs1, graphs2):
        """Prepares the edge label parameters.

        Args:
            graphs1 (list with nx medi graphs): a list with graphs
            graphs2 (list with nx medi graphs): a list with graphs

        Returns:
            None
        """

        es = [self.get_edge_labels(g) for g in graphs1]
        es += [self.get_edge_labels(g) for g in graphs2]
        edge_labels = []
        dic = {}
        for el in es:
            for label in el:
                if label not in dic:
                    edge_labels.append(label)
                    dic[label] = True
        #edge_labels = Counter(alles).most_common()
        #edge_labels = [x for x, count in edge_labels]
        param = self.sample_edge_label_param(n=len(edge_labels))
        self.params = param
        self.param_keys = {edge_labels[idx]:idx for idx in range(len(edge_labels))}
        return None

    def get_edge_labels(self, G):
        """Retrieve all edge labels from a graph

        Args:
            G (nx medi graph): nx multi edge dir. graph
        Returns:
            list with edge labels
        """

        out = []
        for (n1, n2, _) in G.edges:
            edat = G.get_edge_data(n1, n2)
            for k in edat:
                label = edat[k]["label"]
                out.append(label)
        return out

    def sample_edge_label_param(self, n=1):
        """initialize edge parameters. 
        
        The idea with min entropy 
        is to better be able distinguish between edges. This helps with 
        label discirmintation in ARG tasks (but slightly reduces performance
        in other tasks, for other tasks similar or learnt edge weights may 
        be better)

        Args:
            n (int): how many parameters are needed?
        
        Returns:
            array with parameters
        """
        
        params = []
        
        if self.init == "random_uniform":
            for _ in range(n):
                params.append(np.random.uniform(0.25, 0.35, size=(1)))
        
        elif self.init == "min_entropy": 
            for _ in range(n):
                sample = []
                for _ in range(10):
                    sample.append(np.random.uniform(0.0, 1, size=(1)))
                entropies = []
                for i in range(10):
                    entropies.append(
                            entropy(np.array(np.array(params).flatten() 
                                        + list(sample[i])).flatten()))
                argmin = np.argmin(entropies)
                params.append(sample[argmin])
        
        return np.array(params)
            

class AmrWasserPredictor(GraphSimilarityPredictor):

    def __init__(self, params=None, param_keys=None, iters=2):
        self.params = params
        if params is None:
            self.params = []
        self.param_keys = param_keys
        if param_keys is None:
            self.param_keys = {}
            self.unk_edge = 0.33
        else:
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
            #labels.append(nx_latent.nodes[node]["label"])
            labels.append(node)
        return np.array(vecs), labels

    def maybe_has_param(self, label):
        """safe retrieval of an edge parameter"""
        if label not in self.param_keys:
            return self.unk_edge
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
            gs1 = self.get_stats(a1, a2, stattype='nodeoccurence')
            gs2 = self.get_stats(a1, a2, stattype='tripleoccurence')
            v1, v2 = gs1[0], gs1[1]
            v1, v2 = np.concatenate([v1, gs2[0]]), np.concatenate([v2, gs2[1]])
            kv = self.wlk(a1, a2, iters=self.iters, kt=self.simfun, init_vecs=(v1, v2), 
                            weighting="linear", stattype="nodeoccurence")
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
        
        newn = [G.nodes[node]["label"]]

        for nb in G.neighbors(node):
            for k in G.get_edge_data(node, nb):
                el = G.get_edge_data(node, nb)[k]['label']
                label = G.nodes[nb]["label"]
                newn.append(el + '_' + label)
        
        for nb in G.predecessors(node):
            for k in G.get_edge_data(nb, node):
                el = G.get_edge_data(nb, node)[k]['label']
                label = G.nodes[nb]["label"]
                newn.append(el + '_' + label)
        
        return newn
    
    def wl_gather_nodes(self, G):
        """apply gathering (wl_gather_node) for all nodes
        
        Args:
            G (nx medi graph): the graph

        Returns:
            a dictionary node -> neigjborhood
        """

        dic = {}
        for node in G.nodes:
            label = G.nodes[node]["label"]
            dic[label] = self.wl_gather_node(node, G)
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
            x1, x2, vocab = self.wl_iter(nx_g1, nx_g2, stattype=stattype)
            v1s.append(x1)
            v2s.append(x2)
            vocabs.append(vocab)
        #print(v1s, v2s, vocabs)
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
        self.update_node_labels(nx_g1, d1)
        self.update_node_labels(nx_g2, d2)
        stats1, stats2, vocab = self.get_stats(nx_g1, nx_g2, stattype=stattype)
        return stats1, stats2, vocab

    def update_node_labels(self, G, dic):
        for node in G.nodes:
            label = G.nodes[node]["label"]
            G.nodes[node]["label"] = dic[label]
        return None


    
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
        
        vec1, vec2, vocab = None, None, None
        if stattype == 'nodecount':
            vec1, vec2, vocab =  self.nc(g1, g2)
        if stattype == 'nodeoccurence':
            v1, v2, voc = self.nc(g1, g2)
            v1[v1>1] = 1
            v2[v2>1] = 1
            vec1, vec2, vocab =  v1, v2, voc
        if stattype == 'triplecount':
            vec1, vec2, vocab = self.tc(g1, g2)
        if stattype == 'tripleoccurence':
            v1, v2, voc = self.tc(g1, g2)
            v1[v1>1] = 1
            v2[v2>1] = 1
            vec1, vec2, vocab =  v1, v2, voc
        
        return vec1, vec2, vocab
    
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
        #print(items)
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
            label = g1.nodes[node]["label"]
            g1bow.append(label)
            if label not in vocab:
                vocab[label] = i
                i += 1
        for node in g2.nodes:
            label = g2.nodes[node]["label"]
            g2bow.append(label)
            if label not in vocab:
                vocab[label] = i
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
                    el = g1.nodes[node]["label"]
                    el += '_' + g1.get_edge_data(node, nb)[k]['label'] 
                    el += '_' + g1.nodes[nb]["label"]
                    g1bow.append(el)
                    if el not in vocab:
                        vocab[el] = i
                        i += 1
        for node in g2.nodes:
            for nb in g2.neighbors(node):
                for k in g2.get_edge_data(node, nb):
                    el = g2.nodes[node]["label"]
                    el += '_' + g2.get_edge_data(node, nb)[k]['label'] 
                    el += '_' + g2.nodes[nb]["label"]
                    g2bow.append(el)
                    if el not in vocab:
                        vocab[el] = i
                        i += 1
 
        vec1 = self.create_fea_vec(g1bow, vocab)
        vec2 = self.create_fea_vec(g2bow, vocab)
        vec1 = vec1
        vec2 = vec2
        #print(g1bow, g2bow)
        return vec1, vec2, vocab
