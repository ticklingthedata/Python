# 1. Introduction to machine learning

## Classification

* Predicting a qualitative response for an observation can be referred to as classifying that observation, since it involves assigning the observation to a category
or class.

## Clustering:

* Division of data into group of "similar" objects.
* The goal of clustering is to determine the intrinsic grouping in a set of unlabeled data.

## Suprevised learning:

* Supervised learning entails learning a mapping between a set of input variables (typically a vector) and an output variable (also called the supervisory signal) and applying this mapping to predict the outputs for unseen data.
* The mapping/relationship discovered is represented in a structure referred to as a model.
* Supervised learning attempts to inferre a function from supervised training data.

## Unsupervised learning:

* Unsupervised learning studies how systems can learn to represent particular input patterns in a way that reflects the statistical structure of the overall collection of input patterns.
* In unsupervised learning, the machine receives inputs but obtains neither supervised target outputs, nor rewards from its environment.
* In a sense, unsupervised learning can be thought of as finding patterns in the data above and beyond what would be considered noise.

## Reinforcement learning:

* Reinforcement learning is the problem of getting an agent to act in the world so as to maximize its rewards.
* The learner is not told which actions to take. But instead must discover which actions yield the most reward by trying them.
* The agent should discover a good policy from its experiences of the environment without losing too much reward along the way.

## Structured prediction:

* Structured prediction is fundamentally a problem of representation, where the representation must capture both the discriminative interactions between x and y and also allowfor efficient combinatorial optimization over y.
* Structured prediction is about predicting structured outputs from input data in contrast to predicting just a single number, like in classification or regression.

## Neural networks:

* The artificial neural network learns by updating the network architecture and connection weights so that the network can efficiently perform a task.
* It can learn either from available training patterns or automatically learn from examples or input-output relations.

## Deep learning:

* Deep learning refers to a rather wide class of machine learning techniques and architectures, with the hallmark of using many layers of non-linear information processing that are hierarchical in nature.


# 2. Classification:

## Discriminant analysis:

* Used to distinguish distinct sets of observations and allocate new observations to previously defined groups.
* The main goals of discriminant analysis are discrimination and classification.

## Multinomial logistic regression:

* Used to predict categorical placement in or the probability of category membership on a dependent variable, based on multiple independent variables.
* It uses maximum likelihood estimation rather than the least squares estimation used in traditional multiple regression.

## Tobit regression:

* Used to describe the relationship between non-negative dependent variables and independent variables.
* Also known as a censored regression model, designed to estimate linear relationships between variables when there is either left or right censoring in the dependent variable.
* This model is for metric dependent variables and then it is limited in the sense that we observe it only if it is above or below some cut off level.

## Poisson regression:

* Similar to regular multiple regression except that the dependent (Y) variable is an observed count that follows the Poisson distribution.
* Deals with situations in which the dependent variable is a count.


# 3. Clustering:

## Hierarchical clustering:

* For a given set of data points, the output is produced in the form of a binary tree (dendrogram).
* In the binary tree, the leaves represent the data points while internal nodes represent nested clusters of various sizes.
* Each object is assigned a separate cluster. Evaluation of all the clusters takes place based on a pairwise distance matrix.
* The distance matrix will be constructed using distance values. The pair of clusters with the shortest distance must be considered.
* The identified pair should then be removed from the matrix and merged together.
* The merged clusters' distance must be evaluated with the other clusters and the distance matrix should be updated.
* The process is to be repeated until the distance matrix is reduced to a single element.

## K-means clustering:

* It's a method for estimating the mean (vectors) of a set of K-groups.This method is unsupervised, non-deterministic, and iterative in nature.
* The method produces a specific number of disjointed, flat (non-hierarchical) clusters. K denotes the number of clusters.
* Each of the clusters has at least one data point. The clusters are non-overlapping and non-hierarchical in nature.
* The dataset is partitioned into K number of clusters. The data points are randomly assigned to each of the clusters.
* If a data point is closest to its own cluster, it is not changed. If a data point is not close to its own cluster, it is moved to the cluster to which it is closest.
* The steps are repeated for all the data points till no data points are moving from one cluster to another. At this point the clusters are stabilized and the clustering process ends.

# 4. Model Selection and Regularization:

## Subset selection:

* The learning algorithm is faced with the problem of selecting some subset of features upon which to focus its attention, while ignoring the rest.
* If there are m variables and the best regression model consists of p variables, p≤m, then a more general approach to pick the best subset might be to try all
possible combinations of p variables and select the model that fits the data the best.

## Shrinkage methods:

* Refer to shrinkage methods of estimation or prediction in regression situations; useful when there is multi co-linearity among the regressors.
* Retain a subset of the predictors, while discarding the rest.
* More continuous and don't suffer as much from high variability (in constrast to Subset selection).

## Dimension reduction methods:

* May be seen as the process of deriving a set of degrees of freedom, which can be used to reproduce most of the variability of a dataset.

# 5. Nonlinearity:

## Generalized additive models

## Smoothing splines
 
## Local regression

# 6. Supervised Learning:

## Decision tree learning:

* A decision tree is a classifier which recursively partitions the instance space or the variable set.
* Decision trees are represented as a tree structure where each node can be classified as either a leaf node or a decision node.
* A leaf node holds the value of the target attribute, while a decision node specifies the rule to be implemented on a single attribute-value.
* Each decision node splits the instance space into two or more sub-spaces according to a certain discrete function of the input attributes-values.
* Each test considers a single attribute, such that the instance space is partitioned according to the attribute's value.
* After implementing the rule on the decision node, a sub-tree is an outcome.
* Each of the leaf nodes holds a probability vector indicating the probability of the target attribute having a certain value.
* Instances are classified by navigating them from the root of the tree down to a leaf, according to the outcome of the tests along the path.

## Naive Bayes:

* It is a linear classifier based on the Bayes' theorem, which states that the presence of a particular feature of a class is unrelated to the presence of any other feature. 
* Bayesian classifiers can predict class membership probabilities such as the probability.
* Bayesian belief networks is joint conditional probability distribution that allows class-conditional independencies to be defined between subsets of variables.

## Random forest:

* Collections of decision trees that provide predictions into thestructure of data.
* They provide variable rankings, missing values, segmentations, and reporting for each record to ensure deep data understanding.
* After each tree is built, all the data is run down the tree. For each of the pairs of cases, vicinities are computed.
* If two cases occupy the same terminal node, their vicinities are increased by one.
* At the end of the run, normalization is carried out by dividing by the number of trees.
* Proximities are used in replacing missing data, locating outliers, and producing to reveal low-dimensional understandings of the data.
* The training data, which is out-of-bag data, is used to estimate classification error and to calculate the importance of variables.
* A random forest is an effective method for estimating missing data, and maintains accuracy when a large proportion of the data is missing.

## Support vector machines (SVM):

* SVMs make use of a (nonlinear) mapping function φ which transforms data in the input space to data in the feature space in such a way as to render a problem linearly separable.
* The SVM then discovers the optimal separating hyperplane which is then mapped back into input space via φ-1.
* Among the possible hyperplanes, we select the one where the distance of the hyperplane from the closest data points (the margin) is as large as possible.


## Stochastic gradient descent:

* Also known as incremental gradient descent, is a stochastic approximation of the gradient descent optimization method for minimizing an objective function that is written as a sum of differentiable functions.
* It tries to find minima or maxima by iteration. As the algorithm sweeps through the training set, it performs the above update for each training example.
* Several passes can be made over the training set until the algorithm converges.
* If this is done, the data can be shuffled for each pass to prevent cycles.

# 7. Unsupervised Learning:

## Self-organizing map (SOM):

* Based on competitive learning, in which output neurons compete amongst themselves to be activated, with the result that only one is activated at any one time.
* The principal goal of a SOM is to transform an incoming arbitrary dimensional signal into a one- or two-dimensional discrete map, and to perform this transformation adaptively in a topologically ordered fashion.


## Vector quantization:

* Quantization is the process of mapping an infinite set of scalar or vector quantities by a finite set of scalar or vector quantities.
* Vector quantization performs quantization over blocks of data, instead of a single scalar value.
* The quantization output is an index value that indicates another data block (vector) from a finite set of vectors, called the codebook.
* The selected vector is usually an approximation of the input data block.
* Reproduction vectors are known as encoders and decoders.
* The encoder takes an input vector, which determines the best representing reproduction vector, and transmits the index of that vector.
* The decoder takes that index and forms the reproduction vector.

# 8. Reinforcement Learning:

## The Markov chain:

* A sequence of trials of an experiment is a Markov chain if the outcome of each experiment is one of the set of discrete states, and the outcome of the experiment is dependent only on the present state and not of any of the past states.
* The probability of changing from one state to another state is called a transition probability.
* The transition probability matrix is an n × n matrix such that each element of the matrix is non-negative and each row of the matrix sums to one.

## Continuous time Markov chains:

* Continuous-time Markov chains can be labeled as transition systems augmented with rates that have discrete states.
* The states have continuous time-steps and the delays are exponentially distributed.
 
## Monte Carlo simulations:

* These simulations are a stochastic simulations of system behavior.
* They use sampling experiments to be performed on the model and then conduct numerical experiments to obtain a statistical understanding of the system behavior.
* They uses random numbers that are uniformly distributed over the interval [0, 1]. These uniformly distributed random numbers are used for the generation of stochastic variables from various probability distributions.
* Sampling experiments are then generated, which are associated with the modeling of system behavior.

# 9. Structured Prediction:

## Hidden Markov model (HMM):

* It's s statistical method of characterizing the observed data samples of a discrete-time series.
* The data samples in the time series can be discretely or continuously distributed. They can be scalars or vectors.
* The underlying assumption of the HMM is the the data samples can be well characterized as a parametric random process, and the parameters of the stochastic process can be estimated in a precise and well-defined framework.


# 10. Neural Networks:

* A neural network is a concept involving the weights and connections between neurones.
* Data is transferred between neurons via connections, with the connecting weight being either excitatory or inhibitory.

