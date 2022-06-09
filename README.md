# Gini Estimation

This is an 1-day experiment to assess the coverage of Confidence Intervals for normalized Gini index and ROC AUCs.

Existing published papers on this matter consider either small sample sizes, omit any experiment on CI coverage, or suppose very specific distributions.

This analysis was focused on *my specific case*, usually measuring those metrics in an imbalanced dataset, with sizes varying from 10,000 to 10,000,000 points.

Two CI methods are implemented: normal distribution and Balaswamy-Vardhan (2015).

It seems that even for medium-sized samples of 100,000 observations, the distribution of Gini index looks like a normal but it is not quite there.
In several cases, the confidence interval of 99% that supposes normal distribution covers only 95%.
See the notebook.
