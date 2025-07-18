# Taxonomy Evaluation Metric Calculations

The following folders contain the evaluations of the laconicity, lucidity, completeness, and soundness, which have been provided by Kaplan et al. [1] and is available [online](https://github.com/Eden-06/abstraction-quality/). 

Because we have multiple iterations of our taxonomy, we provide the calculations for each iteration. Each iteration has its own subfolder.

Within each subfolder, you will find the corresponding `catalog_taxonomy.txt`, `extractions.txt`, and `mapping.txt` files for that iteration. These files have been generated in the `taxonomy_evaluation` notebook located in the parent folder. The results of the calculations are stored in the `results.txt` file.


Start the calculation with the following command:

```
ruby aquality.rb catalog_taxonomy.txt extractions.txt mapping.txt > results.txt
```


[1] Kaplan, A., et al.: Introducing an evaluation method for taxonomies. In: EASE. ACM (2022), accepted, to appear

