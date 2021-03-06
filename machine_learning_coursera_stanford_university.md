# Machine learning coursera stanford university

a:=b is assignment
a=b is truth assertion

alpha is learning rate

gradient descent algorithn
-use simultaneous update
-when approaching local min, gradient descent automatically takes smaller steps even at same learning rate.

## vector
nx1 matrix
usually 1-indexed (eg y1,y2,..) instead of 0-indexed (eg y0,t1,...)

## fitting linear regression
gradient descent VS normal equation
-if number of features is small, use normal equation
-no need feature scaling if using normal equation method compared to gradient descent. magnitude of feature values insignificant compared to computational cost
-When implementing the normal equation in octave we want to use the 'pinv' function rather than 'inv.'
(Xtranpose X) may be noninvertible. The common causes are:
Redundant features, where two features are very closely related (i.e. they are linearly dependent)
Too many features (e.g. m ≤ n). In this case, delete some features or use "regularization" 
However, if you add regularization to normal equations, (Xtranpose X + λ⋅L) becomes invertible even when m ≤ n.

applying linear regression to classification problem not a good idea:
-if have extreme values, will cause bad classification
-even if all values are 0 or 1, the hypothesis output can end up being not 0 or 1.

logistic regression is a classification algorithm even though name is regression.
will restrict hypothesis output to between 0 and 1

logistic function=sigmoid function
asymptote at 0 and 1.

dont have to plot training set to plot decision boundary.hypothesis function g(z) is not dependent on training set
decision boundary is created by the hypothesis function

for discrete classification, ouput of hypothesis >=0.5 if y=1 and vice versa.
predict "y=1" if input (z) of logisitic function  z>=0 for its output >=0.5.

cost function of linear regression: sum of squared diff
problem of using the same cost function is if were to use sigmoid/logistic function, the cost function can turn out non-convex. there is possibility global min cannot be found cause too many local minimmum.
hence must use the new type of cost function for logistic regression to guarantee that J(θ) is convex for logistic regression.
J(θ)=1/m∑i=1mCost(hθ(x(i)),y(i))
Cost(hθ(x),y)=−log(hθ(x))if y = 1
Cost(hθ(x),y)=−log(1−hθ(x))if y = 0

Cost(hθ(x),y)=0 if hθ(x)=y
Cost(hθ(x),y)→∞ if y=0andhθ(x)→1
Cost(hθ(x),y)→∞ if y=1andhθ(x)→0
summary meant: 
If our correct answer 'y' is 0, then the cost function will be 0 if our hypothesis function also outputs 0. If our hypothesis approaches 1, then the cost function will approach infinity.
If our correct answer 'y' is 1, then the cost function will be 0 if our hypothesis function outputs 1. If our hypothesis approaches 0, then the cost function will approach infinity.
to find the theta that minimizes the cost function for logisitic regression, use gradient descent.
however, not that the gradient descent is not the same as linear regression even though the update algorithm looks exactly same. because the hypothesis function hθ(x) is not the same. previously is θ^T(x). Now is 1/(1+e^θ^T(x)).

feature scaling help gradient descent converge much faster for linear regression
same can be applied for logistic regression
Feature scaling involves dividing the input values by the range (i.e.the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1. 
Mean normalization involves subtracting the average value for an input variable from the values for that input variable, resulting in a new average value for the input variable of just 0.

Debugging gradient descent. Make a plot with number of iterations on the x-axis. Now plot the cost function, J(θ) over the number of iterations of gradient descent. If J(θ) ever increases, then you probably need to decrease α.

Automatic convergence test. Declare convergence if J(θ) decreases by less than E in one iteration, where E is some small value such as 10−3.
However in practice it's difficult to choose this threshold value.

## optimization algorithm:
-gradient descent
-conjugate gradient (adv numerical computing topics)
-BFGS(adv numerical computing topics)
-L-BFGS(adv numerical computing topics)
pros: no need pick learning rate. faster than gradient descent. 
cons: more complex
use algorithms for these provided by Octave already. they are better at optimizing theta.

can use one-vs-all method to solve multiclasss classification

overfitting problem that can arise from learning algorithm applied to certain machine learning applications
solutions are:
-reduce number of features (manually select/Use a model selection algorithm)
-regularization (dont penalize theta 0 for linear regression). It keeps all the features, but reduce the parameters. Regularization works well when we have a lot of slightly useful features.
can apply regularization to both linear regression and logistic regression, and also normal equations.

The λ, or lambda, is the regularization parameter. It determines how much the costs of our theta parameters are inflated.

regularization parameter lambda if too big, can lead to underfitting also even though it's goal was to control the number of features. can end up with too little feature. also meant too high bias.
 
regularization ensures X transpose X matrix is non-invertible (eg when m<=n) as long as lambda >0. 

vectorized implementation usually more efficient than non-vetorized.

When normalizing the features, it is important to store the values used for normalization - the mean value and the stan-
dard deviation used for the computations. After learning the parameters from the model, we often want to predict new y values given new x value, we must normalize x using the mean and standard deviation that we had previously computed from the training set

A logistic regression classifier trained on this higher-dimension feature vector will have a more complex decision boundary and will appear nonlinear when drawn in a 2-dimensional plot.
While the feature mapping allows us to build a more expressive classifier, it also more susceptible to overfitting.

regularization is only for parameter 0 1 and onwards. theta0 do not regularize.
note that for regularized logistic regression, the regularization term excludes j=0. it starts from j=1 to j=n.

smaller theta means simplier hypothetsis, hence less prone to overfitting.

in neral network, sometimes "parameters"="weights"
x0 input node is sometimes called the "bias unit."

use neutral network to get complex non-linear hypothesis

sigmoid(logistic) activiation function =g(z)

Example: If layer 1 has 2 input nodes and layer 2 has 4 activation nodes. Dimension of Θ(1) is going to be 4×3 where s_j=2 and s_{j+1}=4, so  s_{j+1}*(s_j + 1) = 4*3

hidden layers in neural network can help compute more complex features to feed into the final output layer and have more complex hypotheses.

single neuron in neural network can create logical functions (AND, OR)

XNOR: x1=x2=0 and x1=x2=1

The outputs of a neural network are not probabilities, so their sum need not be 1.
