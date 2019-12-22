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

# 6. Machine Learning:

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




