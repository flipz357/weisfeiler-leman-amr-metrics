import argparse
import numpy as np
import json

def build_arg_parser():

    parser = argparse.ArgumentParser(
            description='Argument parser for WWLK')

    parser.add_argument('-a'
            , type=str
            , help='file path to first SemBank')

    parser.add_argument('-b'
            , type=str
            , help='file path to second SemBank')
    
    parser.add_argument('-w2v_uri'
            , type=str
            , default='glove-wiki-gigaword-100'
            , help='gensim uri for word embeddings, see \
                    https://github.com/RaRe-Technologies/gensim-data#models')
    
    parser.add_argument('-k'
            , type=int
            , nargs='?'
            , default=2
            , help='number of WL iterations')
    
    parser.add_argument('-log_level'
            , type=int
            , nargs='?'
            , default=30
            , choices=list(range(0,60,10))
            , help='logging level (int), see\
                    https://docs.python.org/3/library/logging.html#logging-levels')
    
    parser.add_argument('--edge_to_node_transform'
            , action='store_true'
            , help='trasnform to equivalent unlabeled-edge graph, e.g.,\
                    (1, :var, 2) -> (1, :edge, 3), (3, :edge, 2), 3 has label :var')
    
    parser.add_argument('-random_init_relation'
            , type=str
            , default='min_entropy'
            , choices=['min_entropy', 'random_uniform', 'ones', 'constant']
            , help='how to initialize relation embeddings')
        
    parser.add_argument('-output_type'
            , type=str
            , default='score'
            , choices=['score', 'score_corpus', 'score_alignment']
            , help='output options:\
                    score: one score per line for every input graph pair\
                    score_corpus: average score\
                    score_alignment: same as "score" but also provide alignment')
    
    parser.add_argument('-communication_direction'
            , type=str
            , default='both'
            , choices=['both', 'fromout', 'fromin']
            , help='message passing direction:\
                    both: graph is treated as undirected\
                    fromout: node receive info from -> neighbor ("bottom-up AMR")\
                    fromin: node receive info from <- neighbor ("top-down AMR")')

    parser.add_argument('-stability_level'
            , type=int
            , nargs='?'
            , default=0
            , help='compute expected distance matrix via sampling,\
                    increases score stability (random vectors of OOV tokens)\
                    but increases runtime')
    
    parser.add_argument('-round_decimals'
            , type=int
            , nargs='?'
            , default=3
            , help='decimal places to round scores to. Set to large negative number\
                    to prevent any rounding')

    return parser

if __name__ == "__main__":

    import log_helper

    args = build_arg_parser().parse_args()
    logger = log_helper.set_get_logger("Wasserstein AMR similarity", args.log_level)
    logger.info("loading amrs from files {} and {}".format(
        args.a, args.b))
    
    import data_helpers as dh
    import amr_similarity as amrsim
    import graph_helpers as gh

    amrfile1 = args.a
    amrfile2 = args.b
    
    string_amrs1 = dh.read_amr_file(amrfile1)
    graphs1, nodemap1 = gh.parse_string_amrs(string_amrs1, edge_to_node_transform=args.edge_to_node_transform)

    string_amrs2 = dh.read_amr_file(amrfile2)
    graphs2, nodemap2 = gh.parse_string_amrs(string_amrs2, edge_to_node_transform=args.edge_to_node_transform) 

    def get_scores():
        prepro = amrsim.AmrWasserPreProcessor(w2v_uri=args.w2v_uri, init=args.random_init_relation)
        
        predictor = amrsim.AmrWasserPredictor(preprocessor=prepro, iters=args.k, stability=args.stability_level, 
                                                communication_direction=args.communication_direction)
        
        preds = predictor.predict(graphs1, graphs2)
        return preds
    
    def get_scores_alignments():
        prepro = amrsim.AmrWasserPreProcessor(w2v_uri=args.w2v_uri, init=args.random_init_relation)
        
        predictor = amrsim.AmrWasserPredictor(preprocessor=prepro, iters=args.k, stability=args.stability_level)
        
        preds, aligns = predictor.predict_and_align(graphs1, graphs2, nodemap1, nodemap2)
        return preds, aligns
    
    
    if args.output_type == 'score':
        preds = get_scores()
        preds = np.around(preds, args.round_decimals)
        print("\n".join(str(pr) for pr in preds))
        
    elif args.output_type == 'score_corpus':
        preds = get_scores()
        print(np.around(np.mean(preds), args.round_decimals))

    elif args.output_type == 'score_alignment':
        preds, aligns = get_scores_alignments()
        preds = np.around(preds, args.round_decimals)
        jls = [json.dumps({"score":pred, "alignment":aligns[i]}) for i, pred in enumerate(preds)]
        print("\n".join(jls))
            

