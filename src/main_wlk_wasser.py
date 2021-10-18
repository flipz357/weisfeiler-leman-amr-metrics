import argparse
import numpy as np

def build_arg_parser():

    parser = argparse.ArgumentParser(
            description='amr')

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
    
    parser.add_argument('-ensemble_n'
            , type=int
            , nargs='?'
            , default=1
            , help='calculate similarity multiple times with different \
                    edge random intitializations.')

    parser.add_argument('-log_level'
            , type=int
            , nargs='?'
            , default=30
            , choices=list(range(0,60,10))
            , help='logging level (int), see\
                    https://docs.python.org/3/library/logging.html#logging-levels')
    
    parser.add_argument('-random_init_relation'
            , type=str
            , default='min_entropy'
            , choices=['min_entropy', 'random_uniform']
            , help='how to initialize relation embeddings')
    
    parser.add_argument('--corpus_score'
            , action='store_true'
            , help='output only average score over sentences')

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
    
    amrs_in1 = dh.read_amr_file(amrfile1)
    amrs1 = [gh.stringamr2graph(s) for s in amrs_in1]
    triples1 = [gh.graph2triples(G) for G in amrs1]
    graphs1 = [gh.amrtriples2nxmedigraph(tx) for tx in triples1]
    graphs1 = [elm[0] for elm in graphs1]
    amrs_in2 = dh.read_amr_file(amrfile2)
    amrs2 = [gh.stringamr2graph(s) for s in amrs_in2]
    triples2 = [gh.graph2triples(G) for G in amrs2]
    graphs2 = [gh.amrtriples2nxmedigraph(tx) for tx in triples2]
    graphs2 = [elm[0] for elm in graphs2]
    
    predss = []
    for _ in range(args.ensemble_n):
        prepro = amrsim.AmrWasserPreProcessor(w2v_uri=args.w2v_uri, init=args.random_init_relation)
        prepro.prepare(graphs1, graphs2)

        prepro.transform(graphs1, graphs2)
        predictor = amrsim.AmrWasserPredictor(params=prepro.params
                                                , param_keys=prepro.param_keys
                                                , iters=args.k)
        preds = predictor.predict(graphs1, graphs2)
        predss.append(preds)
    predss = np.mean(predss, axis=0)
    if args.corpus_score:
        print(np.mean(predss))
    else:
        print("\n".join(str(pr) for pr in predss))

