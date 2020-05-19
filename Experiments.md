
# Experiments

Four main experiments were conducted this semester: a performance comparison between the original model and the Cluster-GCN based model, a scalability comparison between the original and Cluster-GCN models, a performance comparison between the original model and 2-step/3-step neighbor variants, and an analysis of the distribution of features among nodes that were classified correctly and incorrectly. "Cluster-GCN model" refers to a variant of the original model that uses a partitioning method to convert the initial graph into a subgraph before being inputted into the autoencoder (a design motivated by scalability reasons), and "n-step neighbor" refers to a variant of the original model that converts the original adjacency matrix into its n-step neighbor version before being inputted into the autoencoder.

## Original and Cluster-GCN Performance
The original and Cluster-GCN models were trained and tested on the DBLP dataset, then evaluated through several classification metrics. Several variants of the Cluster-GCN model were used with different hyperparameters, including variants where the Cluster-GCN partitioning was used in the training phase but not the testing phase (denoted as "Original Test").

|  | Accuracy | Recall | Precision | F1 | TNR |
|--|--|--|--|--|--|
| Original Model | 0.965 | 0.5 | 0.021 | 0.041 | 0.965
| p=50, q=5, Original Test | 0.965 | 0.483 | 0.021 | 0.04 | 0.965 |
| p=50, q=5, k=10 | 0.993 | 0.143| 0.02| 0.074| 0.994 |
| p=50, q=5, k=30 | 0.98| 0 | 0 | 0 | 0.982 |
| p=50, q=5, k=100 | 0.942| 0 | 0 | 0 | 0.944 |
| p=50, q=1, Original Test | 0.964 | 0.416 | 0.017 | 0.034 | 0.965 |

At a glance, it seems that while using Cluster-GCN partitioning in the training phase alone leads to similar performance as the original method, using partitioning in the testing phase leads to much lower recall, but additional tests are needed to make any firm conclusions. In particular, these tests should be replicated with additional datasets as the DBLP testing set may have had too few anomalies to accurately measure performance. In addition, more combinations of Cluster-GCN hyperparameters could be tested.

## Original and Cluster-GCN Scalability
The original and Cluster-GCN models were each trained on zero matrices of increasing sizes in order to determine how the training time of each would scale to very large graphs. Test time was not considered. Additionally, the Cluster-GCN model was evaluated with random matrices to determine differences in scalability between sparse and dense graphs.

![](https://lh3.googleusercontent.com/vAl9uNw06C-1KMqd51nCA4N3Me2ZyzcZmUo9qfmSlbYxHX2nS01IFQmDpTAH9TH96OwzoVQHiMXZwAV9reo0nkJhMXTzWB_ThgAIzDxU2YnZI1ZYOu85k5qv5P9bap6b6rJeD_4RbxIjaqMWezTuBQ8F6SbBg0gcb7QlnzABVwKxavW77nxl0QVb-nHt7aMBGtJHmGtNNA7X50MGUjObrUGH6hpDvNREhGexhGQdkXf4OkKLMKodXenhv-HlqFzx2hViwXpkWOJKqreU_YtENhh7LLB8dxA9wfYLQI6_2lD5HLj-4uEtZ4vQX7O73g23ax9Prx4vwYOJ4WrtR8i895-mIaVQALIGjvg45chjKef25h4OkA8g9N_DTImg61f3G9a6tYqIosSOZqtS_oHvpkHKz5RLvFP-dFXwdvkQ4pfJZr2wCbDg9OYd22-bfNLis3mTe6gPRyBVn7q279yNegFjq83Lxp-Gm5CY7LO0VNSKviYL-qMm0QyIc2WgGFq_m6m_KXeLjSHq70Jx6XxI6T_8nKqgmpHs4eyXrIr-4149_5KTcUsM9bYa26d-mlWUzY-gwaa1F1PhGtH1kWjVR8pT-Ak4H7hyqsO5mdeS3QLEhlNnAC46jUTVsdpcN6lFrRmNezVQ5Xye8DdPTACpktbiZvDXZ-lE2f0K5l6SBT2KQpKZuhvEparj7H3gAQ=w300-h186-no?authuser=1)
![](https://lh3.googleusercontent.com/rUhouZuDXSUVUOk9dWtYE9hPPtuD4ZD84GvJnUk68IrdvKNYq-vwBDCSUzm93CfBmod0cPEnYKAPShKVfA6veYPaEknSF43ra1CiJgAZu0qYIHZAyUQg90HX4XNyZT7kuyKVBM15xAi_MG_ZkjQwIBxivbrzKBAZdYtwZJDpFk5Vje8Q0rF9f_nEH8SGyt8VLE9bQ07Yw1ejG2CKAJ6Qp47g-MtrwvzE4J79gm-SGJIXwI6C-yOKXZV0pylgqI5Lpmztra2wz4wf-GtyimD6uY5mYGuVsWuHnMyO8rF14QmrJDBVwwkjIRAvbKWcuOHd92Uz6NLHFHsrwf-QNfxrklY2MLPgFRMdPFTwhJYU9N6CJ7b5Djy0yH72tQD8-keO-7AsWLisqX_IddgUrWASO9S41n8m-T3XLtUiXIJ9k0QnIpWW1V7-vC6Bg1-spClwKctAIFuUFhiIWPcOYgEdLVT9j4wSjTevbW8I-HCdx0M2NcnHZvJvQOisiDj1MH2IxPnsPSvPz3hJimIuK53FFFM666I1sy31jAkWRvQnMk_k3kvkBBSn4wSkG3f0wPhu9Jw3CCvOR-dHg278lMaezFASlrmTtYgDV07wLSTyJa0BGd-ZDaEUacAHhi2KRg87a2izPWmY9qUMs-dwiSVpXHsMlVo--BQx8QY2cckS7NtGq2N669fY1VcFd1tCuw=w300-h186-no?authuser=1)
![](https://lh3.googleusercontent.com/5WiOrH5Y0KBUZOXPi4g9ZIfqZesgToDP87Vs3i8H6V5_vw2-Tp7pxZ2f3NGpIiH6WV1TlWdCOhV3PkLbWgr7D8EE-lB0xKjJpVnhZkeXYx0_uRTCiXY6cqE6WTufFOtYIPnI4aGi6FpKkyVSDgGv5EXGLTTXxei6S0kU7MfJx-W3FbOw_CVDiYJxZjtMBs9qPEGUOre6W3d_KXIn3_h-vo6t_WNNAJj3ATIXYDK4NTumWakz0rJjHaccCaZHBxaiULz3MnxthljaY2aWV47zGRH7hpXpCqp-d3nwsLWDyCIkOXKpLHSA6SPkDNKOuHI6djDzXC12EsK6MfDm_tU2gUYPpyhXqsdvPSh7ovlxqq45AxIaqK8TQXwGUaCZhE3LO2yj-I5E2_oup43ayjtR4B0DKhfLVwP8MIc6ScuLtvqYzcCg7vUOQsr_Ofl9rCOWwdgSBPxyudHZXKe7nvKRJzzMjoEUQu1hSwhWb1GxyZc6dNRHGguAif8nAJEr_2wfPsIQZO7RNqDDT7gUQPpmK03flfZ9d11rU2E-2qCl5PvymUwJ32afahfQBwNkkiDJj-tSJ4jjXXy6MiPr1SzlLB5iwVOjP5HeX1PiD0i3MxXvMoA6CZv0a1YrosGKOtqQVdWCOTRjQH-D0Mn1LPq2VrcYtXsSsYOSARA2L7aUHePs_TMQ6rJmiYcvmWinRA=w300-h186-no?authuser=1)
The runtime of the original model increases rapidly with larger graphs (nearly O(n<sup>2.8</sup>) for n x n matrices) while the Cluster-GCN model scales much better (about O(n<sup>2.2</sup>)). The Cluster-GCN model does take longer on dense matrices, but the actual rate of growth in runtime seems to be similar to the sparse case.

## n-Step Neighbor Performance Comparison
The original model and its 2-neighbor and 3-neighbor variants were trained and tested on the UNSW dataset, then evaluated through several classification metrics. In this experiment, classification metrics were reported for graphs at time t, t-1, and t-2 separately rather than in aggregate as previously done.
|  | Accuracy | Recall | Precision | F1 | TNR |
|--|--|--|--|--|--|
|Original model (time t-2)|0.914|0.25|0.5|0.333|0.976|
|Original model (time t-1)|0.883|0.125|0.2|0.153|0.953|
|Original model (time t)|0.862|0.25|0.222|0.235|0.919|
|2-step neighbor (time t-2)|0.894|0.25|0.333|0.286|0.953|
|2-step neighbor (time t-1)|0.894|0.125|0.25|0.167|0.965|
|2-step neighbor (time t)|0.851|0.125|0.125|0.125|0.918|
|3-step neighbor (time t-2)|0.926|0.25|0.667|0.364|0.988|
|3-step neighbor (time t-1)|0.862|0.125|0.143|0.133|0.930|
|3-step neighbor (time t)|0.872|0.25|0.25|0.25|0.930|

Overall, it seems using 2-step or 3-step neighbor variants does not  lead to improved performance over the original model. Thus, other methods for improving the original model's classification will need to be explored.

## Feature Distribution Analysis
The original model was trained and tested on the UNSW dataset, then nodes in the testing set were grouped based on how they were classified and their features were averaged. In addition to the standard features used for the UNSW dataset, an additional feature "num_adj" was included representing the number of connections from the node to others.

|  | ct_dst_sport_ltm | tcprtt | dwin | ct_src_dport_ltm | ct_dst_src_ltm | ct_dst_ltm | smeansz | num_adj | 
|--|--|--|--|--|--|--|--|--|
|Total Positive|3.71|0.14|53.13|2.29|4.13|3.71|307.17|10.54|
|True Positive|1.4|0.152|0|1|2|1.4|1132|10.8|
|False Negative|4.32|0.131|67.105|2.63|4.68|4.32|90.12|10.47|
|Total Negative|1.32|0.025|118.60|1.37|1.69|1.64|110.81|2.90|
|True Negative|1.33|0.025|113.45|1.39|1.69|1.65|52.31|2.89|
|False Positive|1.08|0.032|215.77|1|1.62|1.46|1213.39|3.08|

Some immediate observations can be made based off of the table. For example, anomalous nodes feature higher values for "tcprtt", "ct_dst_src_ltm", and "num_adj", but lower values for "dwin". Certain features seem more responsible for misclassification than others. For example, false negatives had much lower values for "smeansz" than true positives, and false positives had much higher values for "smeansz" than true negatives. This implies the network learned to classify nodes with high "smeansz" values as anomalous and low "smeansz" values as non-anomalous even when other features like "tcprtt" or "dwin" may have suggested the opposite. Experiments could be ran using different combinations of the initial set of features in order to verify this hypothesis.


