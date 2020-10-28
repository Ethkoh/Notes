*Talks and Workshops*

# AI4IMPACT X DataScience SG: Neural Networks for Forecasting - Tips and Tricks
30 April 2020

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

# Deep Learning for Science School 2019 - Lawrence Berkeley National Lab

## 01 - Introduction to Machine Learning - Brenda Ng
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

## 02 - Introduction to Neural Networks I - Mustafa Mustafa
https://www.youtube.com/watch?v=Wu-faXFJIzw&list=PL20S5EeApOSvfvEyhCPOUzU7zkBcR5-eL&index=2

relu one of the most common activation functions

sigmoid and tanh common used for output layer. sigmoid for probability

What is NN? 
family of parametric,non-linear, and hierarchial representation learning functions which are massively optimized with stochastic gradient descent

what kind of functions can NN approximate?
universal approximate theorem



# NYU Deep Learning Spring 2020 

https://atcold.github.io/pytorch-Deep-Learning/

## Week 1 – Lecture: History, motivation, and evolution of Deep Learning

VGG 19 layers
ResNet 50 layers.

can use deep learning for segmentation not just detection

networks are widely over-parameterized but it doesnt overfit. which goes against statistics textbook
hard to prove in theory yet why it works so well despite more parameters than training examples

svm is like 2 layeer nn. first layer train, second layer classifier.
why deep learning good? dont need kernal function to fit

## Week 1 – Practicum: Classification, linear algebra, and visualisation
https://www.youtube.com/watch?v=5_qrxVq1kvc&list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq&index=2

single layer nn can do scaling,rotation,translation,reflection,shearing


## Week 2 – Lecture: Stochastic gradient descent and backpropagation
https://www.youtube.com/watch?v=d9vdh3b787Y&list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq&index=3


# Blockchain and Cryptocurrencies Workshop
General Assembly 
26 October 2020 9-12pm
Instructor: Emre Surmeli

## blockchain
is a database

unique attribute:
1. secure. once written cannot undo
2. immutable. can only read and write. cannot edit or delete.
3. decentralized
4. public. anyone can query to see its content via pseudo wallets.

visa 20-30k transactions per second
bitcoin is slow 7-10 transactions per second

blockchain is slow but more secure

unprocessed transaction stays in mempool waiting to be added to block

blockchain.com

each block has hash. except first block, subsequent contains its hash and previous hash.


## bitcoin
- released in 2009
- electronic cash system
- author: satoshi nakomoto (person was not found. may be a pseudoname for group of ppl)
- only 21 million bitcoins can exist (estimated all mined by 2049)
- 1 bitcoin = 100,000,000 satoshis
- bitcoin is not fully anonymous. wallet can still be found

## public key cryptography (asymeetric cryptograph)
is a cryptographic system that uses pairs of keys: public keys, which may be disseminated widely, and private keys, which are known only to the owner.

## what are miners?
- to calculate hash of next block
- bookkeepers of bitcoin network
- validate transaction
- make sure consensus among the network (proof of work)

every transaction need to be confirmed by 6 other computers, including the miner.

difficulty of puzzles is variable. more miners puzzle more difficult. algorithm does it authomatically to ensure only 1 block created average every 10 minutes.

merkel root purpose is to shrink size of block. very cool tech

new consensus models are in the midst of work.

digicash collapsed because cannot solve double spending problem.

ethereum is a fork of bitcoin. 

bitcoin cash not asic compatatible unlike ehereum or bitcoin. believe in decentralize. anyone with gpu can mine.

## ethereum
- smart contracts. able to run decentralized applications.
stateofthedapps.com
example:
DAO (decentralized autonomous organization) - a fund on ethereum was craeted to show possibilities. however 150 million was stolen/hacked
- writted in solidarity and viper (programming language)

remix ide by etherium to create smart contracts (eg tokens) with less than 200 lines of code.

etherscan.io : smart contracts search with code

what is an ICO?

token creation have to think the details

now approx 6000 cryptocurrencies

DeFi - decentralized finance. getting popular

eter is currency of ethereum
token - to give value to something and utility. exist in blockchain. sub category of eter more specific to the function, a separate value from eter. utility tokens, security tokens (currencies), basic attention token




