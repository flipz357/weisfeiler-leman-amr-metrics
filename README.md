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

```
cd src
python main_wlk_wasser.py -a <amr_file> -b <amr_file>
```

Return AMR n:m alignment projected to original AMR nodes

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

### Handle randomness in Wasserstein WLK

In current Wasserstein WLK edges (and words not in GloVe) are initialized randomly, 
which can lead to some variation in the predictions. If more stable results are desired 
(e.g., for parsing evaluation), consider using setting fixed edge weights 
or use an ensemble average score with high n, e.g.:

```
cd src
python main_wlk_wasser.py -a <amr_parse_file> -b <amr_gold_file> \
                                    -n_ensemble 50 --corpus_score
```

### More Arguments

There are also more arguments that can be set, e.g., 
use different word embeddings (FastText, word2vec, etc.), 
you can check the options out:

```
cd src
python main_wlk_wasser_alignment.py --help
```

## Version notes

- 0.1: initial release

## Citation

 
