# Stable Bayesian Deep Learning of Koopman Operator

We implement a Bayesian Neural Network on Koopman operator with

- Tensorflow 1.8
- Edward 1.3.5

in Python 2 environment.

# Warning:

At the time when I was writing the code, there is a shift from Python 2 to 3 and 
shifting of Edward to Tensorflow.Probability. Thus, this code for now is only for 
informative purpose. I will come back for an upgrade in the future into Python 3 
environment and make everything in PyTorch.

It will be helpful if you want to know
- how is everything exactly coded to enforce stability constraint of Koopman operator?
- how is SVD-augmented autoencoder built?
- how is Koopman operator is built recurrently? how is the loss function constructed?

## Citation

    @article{pan2020physics,
    title={Physics-informed probabilistic learning of linear embeddings of nonlinear dynamics with guaranteed stability},
    author={Pan, Shaowu and Duraisamy, Karthik},
    journal={SIAM Journal on Applied Dynamical Systems},
    volume={19},
    number={1},
    pages={480--509},
    year={2020},
    publisher={SIAM}
    }
