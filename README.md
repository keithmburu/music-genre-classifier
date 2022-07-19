## CS260 Project: Lab Notebook

Keith 11/18/2021 (2 hours)
- Working on PCA

Matt 11/18/2021 (1 hour)
- Working on csv file parsing

Keith 11/21/2021 (2 hours)
- Started converting continuous features to discrete, creating and evaluating
  feature models

Keith 11/22/2021 (2 hours)
- Debugging conversion of continuous features to discrete
- Applying Naive Bayes for multi-class classification

Matt 11/23/2021 (1.5 hours)

- Created entropy class. On init, generates several useful dictionaries to find
  conditional entropies, info gain, as well as a list of features sorted by
  info gain.

Keith 11/23/2021 (1 hour)

- Debugging Naive Bayes, entropy, testing Naive Bayes with best features

Keith 11/24/2021 (1 hour)

- Working on improving runtime

Matt 11/30/2021 (5 hours)

- Writing logistic regression and tests, and trying to debug all of that. Also found the best continuous features,
  and hardcoded them.

Keith 11/30/2021 (3 hours)

- Debugging logistic regression and testing naive bayes

Matt 12/2/2021 (1 hour)

- Optimizing logistic regression. Fixed(?) some issues with the SGD to get better results.

Matt 12/2/2021 (3 hours)

- Adding confusion matrices, visualization for linear regression. Further refining SGD.

Keith 12/3/2021 (1 hour)

- Fixing naive bayes issues with converted features, setting up hardcoded sorted list of discrete features based on input to reduce entropy computation runtime

Keith 12/4/2021 (3 hours)

- Comparing and plotting naive bayes classification accuracy against different percentages of features, looking into sklearn library implementation for potentially more efficient naive bayes for larger datasets

Matt 12/4/2021 (1.5 hours)

- Created proper visualization tools for understanding the correctness of the logistic
regression model. 

Matt 12/6/2021 (1.5 hour)

- Added some more ways to analyze and visualize data for logistic regression.

Keith 12/7/2021 (3 hours)

- Seaborn visualizations for Naive Bayes
- t-SNE visualization

Keith 12/14/2021 (2 hours)

- Adding comments, file and function headers
- Polishing code style, consistency