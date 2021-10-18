import argparse

def build_arg_parser():

    parser = argparse.ArgumentParser(
            description='amr')

    parser.add_argument('-a_train'
            , type=str
            , help='file path to first train SemBank')

    parser.add_argument('-b_train'
            , type=str
            , help='file path to second train SemBank')
     
    parser.add_argument('-a_dev'
            , type=str
            , help='file path to first dev SemBank')

    parser.add_argument('-b_dev'
            , type=str
            , help='file path to second dev SemBank')
    
    parser.add_argument('-y_dev'
            , type=str
            , nargs="?"
            , default=None
            , help='file path to second dev target score')
    
    parser.add_argument('-y_train'
            , type=str
            , nargs="?"
            , default=None
            , help='file path to train target score')
     
    parser.add_argument('-a_test'
            , type=str
            , help='file path to first test SemBank')

    parser.add_argument('-b_test'
            , type=str
            , help='file path to second test SemBank')

    parser.add_argument('-log_level'
            , type=int
            , nargs='?'
            , default=20
            , choices=list(range(0,60,10))
            , help='logging level (int), see\
                    https://docs.python.org/3/library/logging.html#logging-levels')
    
    parser.add_argument('-w2v_uri'
            , type=str
            , nargs="?"
            , default="glove-wiki-gigaword-100"
            , help='string with w2v uri, see gensim docu, e.g.\
                    \'word2vec-google-news-300\'')

    parser.add_argument('-k'
            , type=int
            , nargs='?'
            , default=2
            , help='number of WL iterations') 
    
    parser.add_argument('-init_lr'
            , type=float
            , nargs='?'
            , default=0.75
            , help='initial learning rate') 
    
    return parser

if __name__ == "__main__":
    import log_helper
    
    args = build_arg_parser().parse_args()
    logger = log_helper.set_get_logger("Wasserstein AMR similarity", args.log_level)
    logger.info("loading amrs from files {} and {}".format(
        args.a_train, args.b_train))
    import black_box_optim as optim
    import data_helpers as dh
    import amr_similarity as amrsim
    import graph_helpers as gh

    amrfile1_train = args.a_train
    amrfile2_train = args.b_train
    
    
    amrfile1_dev = args.a_dev
    amrfile2_dev = args.b_dev
    
    amrfile1_test = args.a_test
    amrfile2_test = args.b_test
    """
    amrs_in1 = dh.read_amr_file(amrfile1)
    amrs1 = [gh.stringamr2graph(s) for s in amrs_in1]
    triples1 = [gh.graph2triples(G) for G in amrs1]

    amrs_in2 = dh.read_amr_file(amrfile2)
    amrs2 = [gh.stringamr2graph(s) for s in amrs_in2]
    triples2 = [gh.graph2triples(G) for G in amrs2]
      
    prepro = amrsim.AmrWasserPreProcessor(w2v_uri=args.w2v_uri)
    prepro.prepare(triples1, triples2)
    amrs1, amrs2 = prepro.transform(triples1, triples2)
    """
    string_amrs1_train = dh.read_amr_file(amrfile1_train)
    graphs1_train, node_map1_train = gh.parse_string_amrs(string_amrs1_train)

    string_amrs2_train = dh.read_amr_file(amrfile2_train)
    graphs2_train, node_map2_train = gh.parse_string_amrs(string_amrs2_train)
    
    prepro = amrsim.AmrWasserPreProcessor(w2v_uri=args.w2v_uri)
    prepro.prepare(graphs1_train, graphs2_train)
    prepro.transform(graphs1_train, graphs2_train)
    
    string_amrs1_dev = dh.read_amr_file(amrfile1_dev)
    graphs1_dev, node_map1_dev = gh.parse_string_amrs(string_amrs1_dev)

    string_amrs2_dev = dh.read_amr_file(amrfile2_dev)
    graphs2_dev, node_map2_dev = gh.parse_string_amrs(string_amrs2_dev)
    
    prepro.transform(graphs1_dev, graphs2_dev)
    
    
    string_amrs1_test = dh.read_amr_file(amrfile1_test)
    graphs1_test, node_map1_test = gh.parse_string_amrs(string_amrs1_test)

    string_amrs2_test = dh.read_amr_file(amrfile2_test)
    graphs2_test, node_map2_test = gh.parse_string_amrs(string_amrs2_test)
    
    prepro.transform(graphs1_test, graphs2_test)
    
    """
    predictor = amrsim.AmrWasserPredictor(params=prepro.params, param_keys=prepro.param_keys, iters=args.k) 
    
    amrs_in1_dev = dh.read_amr_file(amrfile1_dev)
    amrs1_dev = [gh.stringamr2graph(s) for s in amrs_in1_dev]
    triples1_dev = [gh.graph2triples(G) for G in amrs1_dev]
    
    amrs_in2_dev = dh.read_amr_file(amrfile2_dev)
    amrs2_dev = [gh.stringamr2graph(s) for s in amrs_in2_dev]
    triples2_dev = [gh.graph2triples(G) for G in amrs2_dev]
    
    amrs1_dev, amrs2_dev = prepro.transform(triples1_dev, triples2_dev)

    amrs_in1_test = dh.read_amr_file(amrfile1_test)
    amrs1_test = [gh.stringamr2graph(s) for s in amrs_in1_test]
    triples1_test = [gh.graph2triples(G) for G in amrs1_test]
    
    amrs_in2_test = dh.read_amr_file(amrfile2_test)
    amrs2_test = [gh.stringamr2graph(s) for s in amrs_in2_test]
    triples2_test = [gh.graph2triples(G) for G in amrs2_test]
    
    amrs1_test, amrs2_test = prepro.transform(triples1_test, triples2_test)
    """
    predictor = amrsim.AmrWasserPredictor(params=prepro.params, param_keys=prepro.param_keys, iters=args.k) 
    
    # if training and dev targets not exists then it is role confusion 
    # which means targets are 0, 1, 0, 1, 0, 1 ....
    if not args.y_train:
        targets = [0.0, 1.0] * len(string_amrs1_train)
        targets = targets[:len(string_amrs1_train)]
    else:
        targets = dh.read_score_file(args.y_train) 
   
    if not args.y_dev:
        targets_dev = [0.0, 1.0] * len(string_amrs1_dev)
        targets_dev = targets_dev[:len(string_amrs1_dev)]
    else:
        targets_dev = dh.read_score_file(args.y_dev) 
    
    
    optimizer = optim.SPSA(graphs1_train, graphs2_train, predictor
                            , targets, dev_graphs_a=graphs1_dev
                            , dev_graphs_b=graphs2_dev, targets_dev=targets_dev
                            , init_lr=args.init_lr, eval_steps=35)
    
    optimizer.fit()
    preds = predictor.predict(graphs1_test, graphs2_test)

    print("\n".join(str(pr) for pr in preds))
    
