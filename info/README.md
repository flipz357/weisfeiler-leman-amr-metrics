Results on BAMBOO. Resuls for STS, SICK, PARA. And an arithmetic mean and sample weighted arithmetic mean that take results on robustness challenges into account. Sample weighted mean assigns more weight to scores from data sets where more data is available (depending on data size).

| Metric      | STS   | SICK  | PARA  | AMEAN | WMEAN |
|-------------|-------|-------|-------|-------|-------| 
| Smatch      | 58.39 | 59.75 | 41.32 | 51.26 | 52.63 |
| S2match def.| 56.39 | 58.11 | 42.40 | 50.94 | 52.12 |
| S2match     | 58.70 | 60.47 | 42.52 | 52.21 | 53.54 |
| Sema        | 55.90 | 53.32 | 33.43 | 46.29 | 46.45 |
| SemBleu(k3) | 56.49 | 57.76 | 32.47 | 47.50 | 48.81 |
| WLK-k2      | 64.27 | 61.19 | 36.67 | 50.45 | 52.03 |
| WLK-k2      | 65.26 | 61.37 | 36.13 | 50.44 | 52.13 |
| WWLK-k2     | 65.06 | 68.97 | 38.71 | 34.68 | 54.89 |
| WWLK-parser | 64.50 | 66.98 | 38.20 | 45.89 | 53.80 |

Note:

- Smatch: standard Smatch.
- S2match def.: S2match with default HPs `-cutoff: 0.5`, `-diffsense: 0.5`. GloVe Embeddings.
- S2match: S2match with HPs `-cutoff: 0.9`, `-diffsense: 0.95`. GloVe Embeddings.
- Sema: from their repo.
- SemBleu: from their repo.
- WLK-k2: Structural Weisfeiler Leman AMR kernel. Contextualize nodes over iterations, collect graph signatures from iterations, cosine similarity. Call with `python main_wlk.py -a <path1> -b <path2>`
- WWLK-k2: Wasserstein Weisfeiler Leman AMR kernel. Contextualize nodes in latent space over iterations, collect node embeddings, Wasserstein distance. Call with `python m2ain_wlk_wasser.py -a <path1> -b <path2> -stability_level 25`
- WWLK-parser: More stable version for more consistent parsing evaluation results. Transforms edge-labeled AMR graph to equivalent graph without edge labels, where edges are now nodes. Call with: `python main_wlk_wasser.py -a <path1> -b <path2> -stability_level 15 -k 2 --edge_to_node_transform -round_decimals 10 -random_init_relation constant'

