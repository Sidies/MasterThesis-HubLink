This folder documents the clustering phase of the taxonomy construction process. It contains the following files:

1. `cluster_analysis.ipynb`: Contains the analysis of the clustering process in a Jupyter notebook.
2. `cluster_increments.json`: A manually created JSON file documenting the clustering increments.

## How the Clustering Was Done
The clustering process was performed in two iterations. The first iteration clusters the extracted question types, while the second iteration clusters the clusters created in the first iteration.

**Iteration 1:**
For each extracted question type:
1. Determine whether an existing cluster is semantically suitable for the question type.
2. If a suitable cluster exists, assign the question type to that cluster.
3. If no suitable cluster exists, create a new cluster and assign the question type to it.

**Iteration 2:**
For each cluster:
1. Assign a suitable name to the cluster.
2. Provide a suitable description for the cluster.
3. Determine if a parent cluster is semantically suitable for the cluster.
4. If a parent cluster exists, assign the cluster to the parent cluster.
5. If no parent cluster exists, create a new parent cluster and assign the cluster to it.

At the end of iteration 2, assign suitable names and descriptions to the parent clusters.