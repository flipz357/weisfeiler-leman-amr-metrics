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

    parser.add_argument('-log_level'
            , type=int
            , nargs='?'
            , default=40
            , choices=list(range(0,60,10))
            , help='logging level (int), see\
                    https://docs.python.org/3/library/logging.html#logging-levels')
    
    parser.add_argument('-k'
            , type=int
            , nargs='?'
            , default=2
            , help='number of WL iterations')

    return parser

if __name__ == "__main__":

    import log_helper

    args = build_arg_parser().parse_args()
    logger = log_helper.set_get_logger("WLK simple AMR similarity", args.log_level)
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

    amrs_in2 = dh.read_amr_file(amrfile2)
    amrs2 = [gh.stringamr2graph(s) for s in amrs_in2]
    triples2 = [gh.graph2triples(G) for G in amrs2]
    
    pp = amrsim.AmrSymbolicPreprocessor()
    amrs1, amrs2 = pp.transform(triples1, triples2)
    predictor = amrsim.AmrSymbolicPredictor(iters=args.k)

    preds = predictor.predict(amrs1, amrs2)

    print("\n".join(str(pr) for pr in preds))

