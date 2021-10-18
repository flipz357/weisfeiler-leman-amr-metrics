import argparse

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

    parser.add_argument('-log_level'
            , type=int
            , nargs='?'
            , default=20
            , choices=list(range(0,60,10))
            , help='logging level (int), see\
                    https://docs.python.org/3/library/logging.html#logging-levels')
    
    parser.add_argument('-k'
            , type=int
            , nargs='?'
            , default=2
            , help='number of WL iterations')
    
    parser.add_argument('-random_init_relation'
            , type=str
            , default='min_entropy'
            , choices=['min_entropy', 'random_uniform']
            , help='how to initialize relation embeddings')
 
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
    graphs1, node_map1 = gh.parse_string_amrs(string_amrs1)
    
    string_amrs2 = dh.read_amr_file(amrfile2)
    graphs2, node_map2 = gh.parse_string_amrs(string_amrs2)
    

    prepro = amrsim.AmrWasserPreProcessor(w2v_uri=args.w2v_uri)
    prepro.prepare(graphs1, graphs2)
    
    prepro.transform(graphs1, graphs2)
    predictor = amrsim.AmrWasserPredictor(params=prepro.params
                                            , param_keys=prepro.param_keys
                                            , iters=args.k)

    preds, align = predictor.predict_and_align(graphs1, graphs2, node_map1, node_map2)
    
    outdict = {"preds":preds, "align":align}
    print(outdict)

