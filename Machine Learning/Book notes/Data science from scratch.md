# 0. The Book:

* Link: [Data Science from Scratch: First Principles with Python](https://www.amazon.ca/Data-Science-Scratch-Principles-Python/dp/149190142X)

# 1. Linear Algebra:

## Vectors:

* Abstractly, vectors are objects that can be added together to form new vectors and that can be multiplied by scalars (i.e., numbers), also to form new vectors.

* Concretely, vectors are points in some finite-dimensional space. Although you might not think of your data as vectors, they are often a useful way to represent numeric data.

* For example, if you have the heights, weights, and ages of a large number of people, you can treat your data as three-dimensional vectors [height, weight, age].

* If two vectors v and w are the same length, their sum is just the vector whose first element is v[0] + w[0], whose second element is v[1] + w[1], and so on. (If they’re not the same length, then we’re not allowed to add them.)

* The scalar product is the result of multiplying each element of the vector by a scalar.

* the dot product of two vectors is the sum of their componentwise products.

* If w has magnitude 1, the dot product measures how far the vector v extends in the w direction. For example, if w = [1, 0], then dot(v, w) is just the first component of v. Another way of saying this is that it’s the length of the vector you’d get if you projected v onto w.

* Magnitude of a vector: square root the sum of squares.

* Disctance between 2 vectors: square root of the sum of squares of (vi-wi).

## Matrices:

* A matrix is a two-dimensional collection of numbers.

* If A is a matrix, then A[i][j] is the element in the ith row and the jth column.

* Matrix A shape: A has len(A) rows and len(A[0]) columns.

* If a matrix has n rows and k columns, we will refer to it as an n × k matrix (think of each row of an n × k matrix as a vector of length k, and each column as a vector of length n).

* A Matrix can be used to represent a dataset consisting of multiple vectors, simply by considering each vector as a row of the matrix. 

* A n × k matrix to represent a linear function that maps k-dimensional vectors to n-dimensional vectors.

* Matrices can also be used to represent binary relationships.

# 2. Statistics

## Central Tendencies:

* Mean (or average): which is just the sum of the data divided by its count.

* Median: which is the middle-most value (if the number of data points is odd) or the average of the two middle-most values (if the number of data points is even).

* Mode: or most common value(s).

## Dispersion:

* Range: which is just the difference between the largest and smallest elements.

* Variance: which is the expectation of the squared deviation of a random variable from its mean.

* Standard deviation: which is the a measure of the amount of variation or dispersion of a set of values.

## Correlation:

* Covariance: whereas variance measures how a single variable deviates from its mean, covariance measures how two variables vary in tandem from their means.

* Correlation: which divides out the standard deviations of both variables.

## Simpson’s Paradox:

* Correlations can be misleading when confounding variables are ignored.

## Some Other Correlational Caveats:

* A correlation of zero indicates that there is no linear relationship between the two variables. However, there may be other sorts of relationships.

* Correlation tells you nothing about how large the relationship is.

## Correlation and Causation:

* If x and y are strongly correlated, that might mean that x causes y, that y causes x, that each causes the other, that some third factor causes both, or nothing at all.

# 3. Probability:

Think of probability as a way of quantifying the uncertainty associated with events chosen from some universe of events.

## Dependence and Independence:

* Two events E and F are dependent if knowing something about whether E happens gives us information about whether F happens (and vice versa). Otherwise, they are independent.

* Mathematically, we say that two events E and F are independent if the probability that they both happen is the product of the probabilities that each one happens: P(E,F)=P(E)P(F)

## Conditional Probability:

* Think of this as the probability that E happens, given that we know that F happens.

* If they E and F are not necessarily independent (and if the probability of F is not zero), then we define the probability of E “conditional on F” as: P(E|F)=P(E,F)/P(F)


## Bayes’s Theorem:

* This of this as a way of “reversing” conditional probabilities.

* Used to know the probability of some event E conditional on some other event F occurring, when we only have information about the probability of F conditional on E occurring: 

* Concretely: P(E|F)=P(F|E)P(E)/[P(F|E)P(E)+P(F|¬E)P(¬E)]

## Random Variables:

* A random variable is a variable whose possible values have an associated probability distribution.

* The associated distribution gives the probabilities that the variable realizes each of its possible values.

* Expected value of a random variable: which is the average of its values weighted by their probabilities.

* Random variables can be conditioned on events just as other events can.

## Continuous Distributions:

* Used to model distributions across a continuum of outcomes. For example, the *uniform distribution* puts *equal weight* on all the numbers between 0 and 1.

* Because there are infinitely many numbers between 0 and 1, this means that the weight it assigns to individual points must necessarily be zero.

* For this reason, we represent a continuous distribution with a *probability density function*(PDF) such that the probability of seeing a value in a certain interval equals the integral of the density function over the interval.

* the *cumulative distribution function* (CDF) gives the probability that a random variable is less than or equal to a certain value. 

## The Normal Distribution:

* The normal distribution is the classic bell curve–shaped distribution and is completely determined by two parameters: its mean *μ* (mu) and its standard deviation *σ* (sigma). 

* The mean indicates where the bell is centered, and the standard deviation how “wide” it is.

* When *μ* = 0 and *σ* = 1, it’s called the *standard normal distribution*.

## The Central Limit Theorem:

* It says (in essence) that a random variable defined as the average of a large number of independent and identically distributed random variables is itself approximately normally distributed.

# 4. Hypothesis and Inference:

## Statistical Hypothesis Testing:

* In the classical setup, we have a null hypothesis *H0* that represents some default position, and some alternative hypothesis *H1* that we’d like to compare it with. 

* Significance: the willingless to make a type 1 error (“false positive”), in which *H0* is rejected even though it’s true. For reasons lost to the annals of history, this willingness is often set at 5% or 1%.

* Power of a test: which is the probability of not making a type 2 error (“false negative”), in which *H0* is not rejected even though it’s false.

## p-Values:

* We compute the probability—assuming *H0* is true—that we would see a value at least as extreme as the one we actually observed.

* If the p-value is greater than the significance, the null is not rejected.

## Confidence Intervals:

* We’ve been testing hypotheses about the value of the event probability *p*, which is a parameter of the unknown distribution. When this is the case, a third approach is to construct a confidence interval around the observed value of the parameter.

* A condifence interval is the answer to the question: How confident can we be about this estimate of the value of *p*?

## p-Hacking:

* If you’re setting out to find “significant” results, you usually can. Test enough hypotheses against your dataset, and one of them will almost certainly appear significant. Remove the right outliers, and you can probably get your p-value below 0.05.

* This is sometimes called *p-hacking* and is in some ways a consequence of the “inference from p-values framework.”

* If you want to do good science, you should determine your hypotheses before looking at the data, you should clean your data without the hypotheses in mind, and you should keep in mind that p-values are not substitutes for common sense.

## Bayesian Inference:

* An alternative approach to inference (p-Hacking) involves treating the unknown parameters themselves as random variables.

* It starts with a prior distribution for the parameters and then uses the observed data and Bayes’s theorem to get an updated posterior distribution for the parameters.

* Rather than making probability judgments about the tests, you make probability judgments about the parameters.

# 5. Gradient Descent:

The gradient is a fancy word for derivative, or the rate of change of a function. It’s a vector (a direction to move) that:

* Points in the direction of greatest increase of a function (intuition on why).
* Is zero at a local maximum or local minimum (because there is no single direction of increase).

## The Idea Behind Gradient Descent:

* It's a technique to solve optimization problems  (something like “minimizes the error of its predictions” or “maximizes the likelihood of the data.”)

* Generally, it's about maximizing or minimizing functions. That is, we need to find the input that produces the largest (or smallest) possible value.

* If a function has a unique global minimum, this procedure is likely to find it. If a function has multiple (local) minima, this procedure might “find” the wrong one of them, in which case you might rerun the procedure from different starting points. If a function has no minimum, then it’s possible the procedure might go on forever.

## Estimating the Gradient:

* If f is a function of one variable, its derivative at a point x measures how f(x) changes when we make a very small change to x.

* The derivative is defined as the limit of the difference quotients.

* When f is a function of many variables, it has multiple partial derivatives, each indicating how f changes when we make small changes in just one of the input variables.

* We calculate its ith partial derivative by treating it as a function of just its ith variable, holding the other variables fixed.

* A major drawback to this “estimate using difference quotients” approach is that it’s computationally expensive. If v has length n, estimate_gradient has to evaluate f on 2n different inputs. If you’re repeatedly estimating gradients, you’re doing a whole lot of extra work.

## Using the gradient:

* One approach to maximizing a function is to pick a random starting point, compute the gradient, take a small step in the direction of the gradient, and repeat with the new starting point. 

* Similarly, you can try to minimize a function by taking small steps in the opposite direction.

## Choosing the Right Step Size:

* Using a fixed step .

* Gradually shrinking the step size over time.

* At each step, choosing the step size that minimizes the value of the objective function.

## Using Gradient Descent to Fit Models:

* In the usual case, this involves some dataset and some (hypothesized) model for the data that depends (in a differentiable way) on one or more parameters

* It also introduces a loss function that measures how well the model fits our data. (Smaller is better). If we think of the data as being fixed, then the loss function tells us how good or bad any particular model parameters are. 

* This means we can use gradient descent to find the model parameters that make the loss as small as possible. 

* Approach:

	* Start with a random value..
	* Compute the mean of the gradients.
	* Adjust in that direction. 
	* Repeat.

## Minibatch and Stochastic Gradient Descent:

One drawback of the preceding approach is that we had to evaluate the gradients on the entire dataset before we could take a gradient step and update our parameters.

* Alternative: a technique called *minibatch gradient descent*, in which we compute the gradient (and take a gradient step) based on a “minibatch” sampled from the larger dataset.

* Another variation is *stochastic gradient descent*, in which you take gradient steps based on one training example at a time.

* The terminology for the various flavors of gradient descent is not uniform. The “compute the gradient for the whole dataset” approach is often called *batch gradient descent*, and some people say *stochastic gradient descent* when referring to the minibatch version (of which the one-point-at-a-time version is a special case).

# 6. Machine Learning (general concepts):

Data science is mostly turning business problems into data problems and collecting data and understanding data and cleaning data and formatting data, after which machine learning is almost an afterthought.

## Modeling:

* A model is simply a specification of a mathematical (or probabilistic) relationship that exists between different variables.

* We’ll use machine learning to refer to creating and using models that are learned from data. In other contexts this might be called predictive modeling or data mining.

* The goal is to use existing data to develop models that we can use to predict various outcomes for new data. 

* Model categorization:
	* Supervised models: in which there is a set of data labeled with the correct answers to learn from.
	* Unsupervised models: in which there are no such labels.
	* Semisupervised models: in which only some of the data are labeled.
	* Online models: in which the model needs to continuously adjust to newly arriving data.
	* Reinforcement models: in which, after making a series of predictions, the model gets a signal indicating how well it did.

## Overfitting and Underfitting:

* Overfitting: producing a model that performs well on the data you train it on but generalizes poorly to any new data.

* Underfitting: producing a model that doesn’t perform well even on the training data.

* Models that are too complex lead to overfitting and don’t generalize well beyond the data they were trained on. 

* The most fundamental approach to avoid building overcomplex models involves using different data to train the model and to test the model. The simplest way to do this is to split the dataset, so that (for example) two-thirds of it is used to train the model, after which we measure the model’s performance on the remaining third.

* Here are a couple of ways this can go wrong:
	* The first is if there are common patterns in the test and training data that wouldn’t generalize to a larger dataset.
	* A bigger problem is if you use the test/train split not just to judge a model but also to choose from among many models. In such a situation, you should split the data into three parts: a training set for building models, a validation set for choosing among trained models, and a test set for judging the final model.

## Correctness:

* It’s common to look at the combination of precision and recall. 
	* Precision measures how accurate our positive predictions were.
	* Recall measures what fraction of the positives our model identified.
	* Sometimes precision and recall are combined into the *F1 score* (This is the harmonic mean of precision and recall)

* Usually the choice of a model involves a tradeoff between precision and recall:
	* A model that predicts “yes” when it’s even a little bit confident will probably have a high recall but a low precision.
	* A model that predicts “yes” only when it’s extremely confident is likely to have a low recall and a high precision.

* Alternatively, you can think of this as a tradeoff between false positives and false negatives.
	* Saying “yes” too often will give you lots of false positives.
	* Saying “no” too often will give you lots of false negatives.

## The Bias-Variance Tradeoff:

* Another way of thinking about the overfitting problem is as a tradeoff between bias and variance.

* Both are measures of what would happen if you were to retrain your model many times on different sets of training data (from the same larger population).

* Low variance and high bias typically correspond to underfitting.

* Low bias and high variance corresponds to overfitting.

* If a model has high bias (which means it performs poorly even on your training data), one thing to try is adding more features. Going from the degree X to the degree X+1 model can introduce considerable improvement.

* If your a has high variance, you can similarly remove features. But another solution is to obtain more data (if you can).

## Feature Extraction and Selection:

* Features are whatever inputs we provide to your model.

* Feature types:
	* Yes or No (typically encoded 1 and 0).
	* Number. 
	* Discrete set of options.

* The types of features in hand constrain the types of models you can use:
	* The Naive Bayes classifier is suited to yes-or-no features.
	* Regression models require numeric features (which could include dummy variables that are 0s and 1s).
	* Decision trees can deal with numeric or categorical data.

* Depending on the situation, it might be appropriate to distill features down to a handful of important dimensions (as in “Dimensionality Reduction”) and use only that small number of features. 

* It might also be appropriate to use a technique (like regularization, which we’ll look at in “Regularization”) that penalizes models the more features they use.

# 7. k-Nearest Neighbors:

## The Model:

* Nearest neighbors is one of the simplest predictive models there is. It makes no mathematical assumptions, and it doesn’t require any sort of heavy machinery. The only things it requires are:
	* Some notion of distance.
	* An assumption that points that are close to one another are similar.

* Nearest neighbors consciously neglects a lot of information, since the prediction for each new point depends only on the handful of points closest to it.

* What’s more, nearest neighbors is probably not going to help you understand the drivers of whatever phenomenon you’re looking at.

* Pick a number k. Then, when you want to classify some new data point, find the k nearest labeled points and let them vote on the new output.

## The Curse of Dimensionality:

* The k-nearest neighbors algorithm runs into trouble in higher dimensions thanks to the “curse of dimensionality,” which boils down to the fact that high-dimensional spaces are vast. 

* Points in high-dimensional spaces tend not to be close to one another at all. As the number of dimensions increases, the average distance between points increases. But what’s more problematic is the ratio between the closest distance and the average distance. 

* So if you’re trying to use nearest neighbors in higher dimensions, it’s probably a good idea to do some kind of dimensionality reduction first.

# 8. Naive Bayes:

* Bayes’s theorem tells us that the probability of an event A conditional on event B (example: the probability that the message is spam *S* conditional on containing the word bitcoin *B* expressed as: **P(S|B)=[P(B|S)P(S)]/[P(B|S)P(S)+P(B|¬S)P(¬S)**).

* The numerator is the probability that a message is spam and contains bitcoin, while the denominator is just the probability that a message contains bitcoin. Hence, you can think of this calculation as simply representing the proportion of bitcoin messages that are spam.

* Imagine now that we have a vocabulary of many words. The key to Naive Bayes is making the (big) assumption that the presences (or absences) of each word are independent of one another, conditional on a message being spam or not. 

* Intuitively, this assumption means that knowing whether a certain spam message contains the word bitcoin gives you no information about whether that same message contains the word rolex. **This is an extreme assumption** (there’s a reason the technique has naive in its name).

# 9. Simple Linear Regression:

## The Model:

* In particular, you hypothesize that there are constants α (alpha) and β (beta) such that: **yi=βxi+α+εi** where ε is an error term representing the fact that there are other factors not accounted for by this simple model.

* Any choice of alpha and beta gives us a predicted output for each input *xi*. Since we know the actual output *yi*, we can compute the error for each pair.

* What we’d really like to know is the total error over the entire dataset. So instead we add up the squared errors. The least squares solution is to choose the alpha and beta that make the sum of square errors as small as possible.

* The choice of alpha simply says that when we see the average value of the independent variable x, we predict the average value of the dependent variable y.

* The choice of beta means that when the input value increases by standard_deviation(x), the prediction then increases by correlation(x, y) * standard_deviation(y). 

* In the case where x and y are perfectly correlated, a one-standard-deviation increase in x results in a one-standard-deviation-of-y increase in the prediction. 

* When they’re perfectly anticorrelated, the increase in x results in a decrease in the prediction. 

* And when the correlation is 0, beta is 0, which means that changes in x don’t affect the prediction at all.

* A better way to figure out how well we’ve fit the data than staring at the graph is the measure of the coefficient of determination (or **R-squared**), which measures the fraction of the total variation in the dependent variable that is captured by the model.

* R-squared must be at least 0, and at most 1. The higher the number, the better our model fits the data. 

## Maximum Likelihood Estimation:

* Imagine that we have a sample of data v1,...,vn that comes from a distribution that depends on some unknown parameter θ (theta). Under this approach, the most likely θ is the value that maximizes this likelihood function—that is, the value that makes the observed data the most probable. 

* In the case of a continuous distribution, in which we have a probability distribution function rather than a probability mass function, we can do the same thing.

* The likelihood based on the entire dataset is the product of the individual likelihoods, which is largest precisely when alpha and beta are chosen to minimize the sum of squared errors.

* That is, in this case (with these assumptions), minimizing the sum of squared errors is equivalent to maximizing the likelihood of the observed data.

# 10. Multiple Regression:

## The Model:

* each input xi is not a single number but rather a vector of k numbers, xi1, ..., xik. The multiple regression model assumes that: **yi=α+β1xi1+...+βkxik+εi**.
* In multiple regression the vector of parameters is usually called β. We’ll want this to include the constant term as well, which we can achieve by adding a column of 1s to our data.

## Further Assumptions of the Least Squares Model:

* The first is that the columns of x are linearly independent—that there’s no way to write any one as a weighted sum of some of the others. If this assumption fails, it’s impossible to estimate beta.

* The second important assumption is that the columns of x are all uncorrelated with the errors ε. If this fails to be the case, our estimates of beta will be systematically wrong.

## Fitting the Model:

* As in a linear model, beta is choosen to minimize the sum of squared errors. Finding an exact solution is not simple to do by hand, gradient descent is used to make it easier.

* But in practice, you wouldn’t estimate a linear regression using gradient descent; you’d get the exact coefficients using linear algebra techniques.

## Interpreting the model:

* You should think of the coefficients of the model as representing all-else-being-equal estimates of the impacts of each factor.

* When a model does not capture the interactions between the variables, one alternative could be introducing a new variable(s) to achieve more clarity.

* Once we start adding variables, we need to worry about whether their coefficients “matter.” There are no limits to the numbers of products, logs, squares, and higher powers we could add.

## Goodness of Fit:

* Keep in mind, however, that adding new variables to a regression will necessarily increase the R-squared.

* Because of this, in a multiple regression, we also need to look at the standard errors of the coefficients, which measure how certain we are about our estimates of each βi.

* The regression as a whole may fit our data very well, but if some of the independent variables are correlated (or irrelevant), their coefficients might not mean much.

* The typical approach to measuring these errors starts with another assumption—that the errors εi are independent normal random variables with mean 0 and some shared (unknown) standard deviation σ.

* In that case, we can use some linear algebra to find the standard error of each coefficient. The larger it is, the less sure our model is about that coefficient. 

## Standard Errors of Regression Coefficients:

* Bootstraping: choosing n data points with replacement from the data, and run computation on those synthetic datasets.

* We repeatedly take a bootstrap_sample of our data and estimate beta based on that sample: 
	* If the coefficient corresponding to one of the independent variables (say, num_friends) doesn’t vary much across samples, then we can be confident that our estimate is relatively tight. 
	* If the coefficient varies greatly across samples, then we can’t be at all confident in our estimate.

* After that  we can estimate the standard deviation of each coefficient.

* We can use these to test hypotheses such as “does  βi equal 0?” Under the null hypothesis βi=0 (and with our other assumptions about the distribution of εi), the statistic:
**tj=ˆβj/ˆσ** (the estimate of βj divided by the estimate of its standard error follows a t-distribution with “n−k degrees of freedom.”)

* As the degrees of freedom get large, the t-distribution gets closer and closer to a standard normal.

* In more elaborate regression scenarios, you sometimes want to test more elaborate hypotheses about the data, such as “at least one of the  βj is nonzero” or “β1 equals  β2 and β3 equals β4”.  You can do this with an *F-test*.

## Regularization:

* Wrinkles of applying linear regression to large datasets:
	* The more variables you use, the more likely you are to overfit your model to the training set. 
	* The more nonzero coefficients you have, the harder it is to make sense of them. If the goal is to explain some phenomenon, a sparse model with three factors might be more useful than a slightly better model with hundreds.

* Regularization: is an approach in which we add to the error term a penalty that gets larger as beta gets larger. We then minimize the combined error and penalty. The more importance we place on the penalty term, the more we discourage large coefficients.
	* For example, in ridge regression, we add a penalty proportional to the sum of the squares of the beta_i (except that typically we don’t penalize beta_0, the constant term).
	* Another approach is lasso regression. Whereas the ridge penalty shrank the coefficients overall, the lasso penalty tends to force coefficients to be 0, which makes it good for learning sparse models.

# 11. Logistic Regression:

## The Logistic Function:

* As its input gets large and positive, it gets closer and closer to 1. As its input gets large and negative, it gets closer and closer to 0.

* Recall that for linear regression we fit the model by minimizing the sum of squared errors, which ended up choosing the β that maximized the likelihood of the data.

* In the logistic regression, we use gradient descent to maximize the likelihood directly. This means we need to calculate the likelihood function and its gradient: **yi=f(xiβ)+εi**:
	* Given some β, our model says that each yi should equal 1 with probability f(xiβ) and 0 with probability 1−f(xiβ). 
	* it’s actually simpler to maximize the log likelihood. Because log is a strictly increasing function, any beta that maximizes the log likelihood also maximizes the likelihood, and vice versa. 
	* Because gradient descent minimizes things, we can work with the negative log likelihood, since maximizing the likelihood is the same as minimizing its negative.
	* If we assume different data points are independent from one another, the overall likelihood is just the product of the individual likelihoods. That means the overall log likelihood is the sum of the individual log likelihoods.

## Support Vector Machines:

* The idea behind the support vector machine is to find the hyperplane that maximizes the distance to the nearest point in each class (the best separates the classes in the training data).

* We can sometimes get around this by transforming the data into a higher-dimensional space. This is usually called the kernel trick because rather than actually mapping the points into the higher-dimensional space (which could be expensive if there are a lot of points and the mapping is complicated), we can use a “kernel” function to compute dot products in the higher-dimensional space and use those to find a hyperplane.

# 12. Decision Trees:

## Definition of a Decision Tree:

* A decision tree uses a tree structure to represent a number of possible decision paths and an outcome for each path.

* Decision trees have a lot to recommend them:
	* They’re very easy to understand and interpret, and the process by which they reach a prediction is completely transparent. 
	* Unlike the other models we’ve looked at so far, decision trees can easily handle a mix of numeric (e.g., number of legs) and categorical (e.g., delicious/not delicious) attributes and can even classify data for which attributes are missing.

* At the same time: 
	* Finding an “optimal” decision tree for a set of training data is computationally a very hard problem. 
	* More important, it is very easy (and very bad) to build decision trees that are overfitted to the training data, and that don’t generalize well to unseen data.

* Most people divide decision trees into classification trees (which produce categorical outputs) and regression trees (which produce numeric outputs).

## Entropy:

* It is the notion of “how much information”. You have probably heard this term used to mean disorder. We use it to represent the uncertainty associated with data:
	* If all the data points belong to a single class, then there is no real uncertainty, which means we’d like there to be low entropy. 
	* If the data points are evenly spread across the classes, there is a lot of uncertainty and we’d like there to be high entropy. 

* In math terms, if pi is the proportion of data labeled as class ci, we define the entropy as:
**H(S)=−p1log2p1−...−pnlog2pn**

* This means the entropy will be small when every pi is close to 0 or 1 (i.e., when most of the data is in a single class), and it will be larger when many of the pi’s are not close to 0 (i.e., when the data is spread across multiple classes).

## The Entropy of a Partition:

* The entropy of a partition results from partitioning a set of data in a certain way. 

* A partition will have low entropy if it splits the data into subsets that themselves have low entropy (i.e., are highly certain), and high entropy if it contains subsets that (are large and) have high entropy (i.e., are highly uncertain).

## Creating a Decision Tree (A simple approach):

* The tree will consist of decision nodes (which ask a question and direct us differently depending on the answer) and leaf nodes (which give us a prediction).

* Given some labeled data, and a list of attributes to consider branching on:
	* If the data all have the same label, create a leaf node that predicts that label and then stop.
	* If the list of attributes is empty (i.e., there are no more possible questions to ask), create a leaf node that predicts the most common label and then stop.
	* Otherwise, try partitioning the data by each of the attributes.
	* Choose the partition with the lowest partition entropy.
	* Add a decision node based on the chosen attribute.
	* Recur on each partitioned subset using the remaining attributes.

* This is what’s known as a “greedy” algorithm because, at each step, it chooses the most immediately best option. Given a dataset, there may be a better tree with a worse-looking first move. If so, this algorithm won’t find it.

## Random Forests:

* Given how closely decision trees can fit themselves to their training data, it’s not surprising that they have a tendency to overfit.

* One way of avoiding this is a technique called *random forests*, in which we build multiple decision trees and combine their outputs. If they’re classification trees, we might let them vote; if they’re regression trees, we might average their predictions.

* To achieve randomness:
	* Rather than training each tree on all the inputs in the training set, we train each tree on the result of bootstrap_sample(inputs). Since each tree is built using different data, each tree will be different from every other tree. This technique is known as bootstrap aggregating or bagging.
	* A second source of randomness involves changing the way we choose the best_attribute to split on. Rather than looking at all the remaining attributes, we first choose a random subset of them and then split on whichever of those is best. This is an example of a broader technique called *ensemble learning* in which we combine several weak learners (typically high-bias, low-variance models) in order to produce an overall strong model.

# 13. Neural Networks:

## Overview:

* An artificial neural network (or neural network for short) is a predictive model motivated by the way the brain operates. 

* Think of the brain as a collection of neurons wired together. Each neuron looks at the outputs of the other neurons that feed into it, does a calculation, and then either fires (if the calculation exceeds some threshold) or doesn’t (if it doesn’t).

* Artificial neural networks consist of artificial neurons, which perform similar calculations over their inputs. 

* Most neural networks are “black boxes”—inspecting their details doesn’t give you much understanding of how they’re solving a problem. And large neural networks can be difficult to train.

## Feed-Forward Neural Networks:

* This consists of discrete layers of neurons, each connected to the next.

* This typically entails:
	* An input layer (which receives inputs and feeds them forward unchanged).
	* One or more “hidden layers” (each of which consists of neurons that take the outputs of the previous layer, performs some calculation, and passes the result to the next layer).
	* An output layer (which produces the final outputs).

* Each (noninput) neuron has a weight corresponding to each of its inputs and a bias.

* We can represent a neuron simply as a vector of weights whose length is one more than the number of inputs to that neuron (because of the bias weight). 

* Then we can represent a neural network as a list of (noninput) layers, where each layer is just a list of the neurons in that layer. That is, we’ll represent a neural network as a list (layers) of lists (neurons) of vectors (weights).

## Backpropagartion:

* This approach is an algorithm which uses gradient descent or one of its variants:
	* Produce the outputs of all the neurons in the network.	
	* We know the target output, so we can compute a loss that’s the sum of the squared errors.
	* Compute the gradient of this loss as a function of the output neuron’s weights.
	* “Propagate” the gradients and errors backward to compute the gradients with respect to the hidden neurons’ weights.
	* Take a gradient descent step. 
	* Typically we run this algorithm many times for our entire training set until the network converges.

# 14. Deep Learning:

* In many neural network libraries, n-dimensional arrays are referred to as *tensors*.

* Layer: something that knows how to apply some function (*activation function*: sigmoid, tanh, Relu, etc) to its inputs and that knows how to backpropagate.

* *Loss functions* can be applied for optimization (softmax, cross-entropy, etc).

* Dropout: a technique that consists in randomly turn off each neuron (that is, replace its output with 0) with some fixed probability. This means that the network can’t learn to depend on any individual neuron, which seems to help with overfitting.


# 15. Clustering:

## Overview:

* Belongs to the unsupervised set of ML techniques in which we work with completely unlabeled data (or in which our data has labels but we ignore them).

* There is generally no “correct” clustering.

* Furthermore, the clusters won’t label themselves. You’ll have to do that by looking at the data underlying each one.

## The Model:

* One of the simplest clustering methods is k-means, in which the number of clusters k is chosen in advance. After which the goal is to partition the inputs into sets 
S1,...,Sk in a way that minimizes the total sum of squared distances from each point to the mean of its assigned cluster.

* There are a lot of ways to assign n points to k clusters, which means that finding an optimal clustering is a very hard problem. We’ll settle for an iterative algorithm that usually finds a good clustering:
	* Start with a set of k-means, which are points in d-dimensional space.
	* Assign each point to the mean to which it is closest.
	* If no point’s assignment has changed, stop and keep the clusters.
	* If some point’s assignment has changed, recompute the means and return to step 2.

* There are various ways to choose a k. One that’s reasonably easy to understand involves plotting the sum of squared errors (between each point and the mean of its cluster) as a function of k and looking at where the graph “bends”.

## Bottom-Up Hierarchical Clustering:

* An alternative approach to clustering is to “grow” clusters from the bottom up. We can do this in the following way:
	* Make each input its own cluster of one.
	* As long as there are multiple clusters remaining, find the two closest clusters and merge them.

* At the end, we’ll have one giant cluster containing all the inputs. If we keep track of the merge order, we can re-create any number of clusters by unmerging. For example, if we want three clusters, we can just undo the last two merges.

* In order to merge the closest clusters, we need some notion of *the distance between clusters*.

