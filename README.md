# Weisfeiler-Leman Graph Kernels for AMR Graph Similarity

The repository contains python code for metric of AMR graph similarity.

## Requirements

Install the following python packages (with pip or conda):

```
numpy (tested: 1.19.4)
scipy (tested: 1.1.0) 
networkx (tested: 2.5)
gensim (tested: 3.8.3)
penman (tested 1.1.0)
```

## Computing AMR metrics

### Basic Wasserstein AMR similarity

Note that the node labels will get initialized with GloVe vectors, 
it can take a minute to load them. If everything should be randomly intitialzed 
(no loading time), set `-w2v_uri none`.

```
cd src
python main_wlk_wasser.py -a <amr_file> -b <amr_file>
```

### Return AMR n:m alignment projected to original AMR nodes

```
cd src
python main_wlk_wasser_alignment.py -a <amr_file> -b <amr_file>
```

### Learning edge parameters for control

```
cd src
python main_wlk_wasser_optimized.py -a_train <amr_file> -b_train <amr_file> \
                                    -a_dev <amr_file> -b_dev <amr_file> \
                                    -a_test <amr_file> -b_test <amr_file> \
                                    -y_train <target_file> -y_dev <target_file>
```

where `<target_file>` is a file the contains a float per line for which we
optimize the parameters. In the end the script will return predictions for
`-a_test <amr_file>` vs. `b_test <amr_file>`.


### Symbolic AMR similarity

```
cd src
python main_wlk.py -a <amr_file> -b <amr_file>
```

## Tips

### Parsing evaluation

Currently, only `main_wlk.py`, i.e., the symbolic WLK provides deterministic results.
Since in current Wasserstein WLK the edges and words not in GloVe are initialized randomly, 
it can lead to some variation in the predictions. If more stable results for WWLK are desired
, consider setting fixed edge weights or use an ensemble average score with, e.g.:

```
cd src
python main_wlk_wasser.py -a <amr_parse_file> -b <amr_gold_file> \
                          -n_ensemble 5 --corpus_score \
                          -random_init_relation random_uniform
```

### More Arguments

There are also more arguments that can be set, e.g., 
use different word embeddings (FastText, word2vec, etc.), 
you can check the options out:

```
cd src
python main_wlk_wasser.py --help
```

## Version notes

- 0.1: initial release

## Citation

```
@article{10.1162/tacl_a_00435,
    author = {Opitz, Juri and Daza, Angel and Frank, Anette},
    title = "{Weisfeiler-Leman in the Bamboo: Novel AMR Graph Metrics and a Benchmark for AMR Graph Similarity}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {9},
    pages = {1425-1441},
    year = {2021},
    month = {12},
    abstract = "{Several metrics have been proposed for assessing the similarity of (abstract) meaning representations (AMRs), but little is known about how they relate to human similarity ratings. Moreover, the current metrics have complementary strengths and weaknesses: Some emphasize speed, while others make the alignment of graph structures explicit, at the price of a costly alignment step.In this work we propose new Weisfeiler-Leman AMR similarity metrics that unify the strengths of previous metrics, while mitigating their weaknesses. Specifically, our new metrics are able to match contextualized substructures and induce n:m alignments between their nodes. Furthermore, we introduce a Benchmark for AMR Metrics based on Overt Objectives (Bamboo), the first benchmark to support empirical assessment of graph-based MR similarity metrics. Bamboo maximizes the interpretability of results by defining multiple overt objectives that range from sentence similarity objectives to stress tests that probe a metric’s robustness against meaning-altering and meaning- preserving graph transformations. We show the benefits of Bamboo by profiling previous metrics and our own metrics. Results indicate that our novel metrics may serve as a strong baseline for future work.}",
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00435},
    url = {https://doi.org/10.1162/tacl\_a\_00435},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00435/1979290/tacl\_a\_00435.pdf},
}

``` 
