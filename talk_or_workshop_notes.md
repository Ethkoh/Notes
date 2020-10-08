### AI4IMPACT X DataScience SG: Neural Networks for Forecasting - Tips and Tricks
### 30 April 2020

MAE vs RMSE
RMSE if not much outliers or shocks. 

using other features such as day or time for time series?
day of week is bad. is 1 to 7 but 1 to 7 is suppose to be side by side.
day of year not too bad.
time is bad. think of angle around circle.
instead of using one number for cyclic data, think of using uv inputs (2 values).

Can you elaborate a little on how you normalize your timeseries? Are all values relative to T0 or would you use other baselines such as moving averages?
random boost or xgboost dont need normalize but very impt for neural network
why need normalize?
-saturation
-inputs need to be about same size. otherwise if take dotproduct will dorminate other numbers

neural network tip
-difference network: add y(T+0) to output.
why? beats persistence, no need detrending, spikes are ok
-momentum and force inputs.
first and second order differences of all inputs as new input 
-input scaling subnet 
symmetric squashing function (tanh)
-dimension reduction subnet
NN dont work well with large dimensional inputs(>100). reduce dimensions using dim reduction/compression
-squared inputs
have properties of both perceptrons and rbf neurons

### Deep Learning for Science School 2019 - Lawrence Berkeley National Lab
### 01 - Introduction to Machine Learning - Brenda Ng
https://www.youtube.com/watch?v=KpDV_FBCWj0&list=PL20S5EeApOSvfvEyhCPOUzU7zkBcR5-eL

three main branches of ML:
-supervised learning
-unsupervised learning
-reinforcement learning

encorder (h=f(x)): compresses the input(x) into a latent feature(h)
decoder (r=g(h)): reconstructs the input(h) from the latent feature(h)
autoencorder: g(f(x))=r, where want r to be as close as x 

word2vec: 
-shallow NN that encodes each word from one-hot sparse vector to a dense vector
unlike autoencoder, word2vec's unsupervised learning is not to reconstruct the input words, but to predict neighbouring words
-encoding center words and context words

continous bag-of-words(CBOW) is faster to train than Skip-Gram(SG) and yields slightly better accuracy for frequent words

reinforcement learning: DL models have been applied to learn policy mappings, value functions and dynamic models

### 02 - Introduction to Neural Networks I - Mustafa Mustafa
https://www.youtube.com/watch?v=Wu-faXFJIzw&list=PL20S5EeApOSvfvEyhCPOUzU7zkBcR5-eL&index=2

relu one of the most common activation functions

sigmoid and tanh common used for output layer. sigmoid for probability

What is NN? 
family of parametric,non-linear, and hierarchial representation learning functions which are massively optimized with stochastic gradient descent

what kind of functions can NN approximate?
universal approximate theorem



# NYU Deep Learning Spring 2020 

https://atcold.github.io/pytorch-Deep-Learning/

### Week 1 – Lecture: History, motivation, and evolution of Deep Learning

VGG 19 layers
ResNet 50 layers.

can use deep learning for segmentation not just detection

networks are widely over-parameterized but it doesnt overfit. which goes against statistics textbook
hard to prove in theory yet why it works so well despite more parameters than training examples

svm is like 2 layeer nn. first layer train, second layer classifier.
why deep learning good? dont need kernal function to fit

### Week 1 – Practicum: Classification, linear algebra, and visualisation
https://www.youtube.com/watch?v=5_qrxVq1kvc&list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq&index=2

single layer nn can do scaling,rotation,translation,reflection,shearing


### Week 2 – Lecture: Stochastic gradient descent and backpropagation
https://www.youtube.com/watch?v=d9vdh3b787Y&list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq&index=3
