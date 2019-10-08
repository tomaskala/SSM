* [Parameter Estimation in Hidden Markov Models With Intractable Likelihoods Using Sequential Monte Carlo](https://www.tandfonline.com/doi/pdf/10.1080/10618600.2014.938811?needAccess=true)
    * Section 4.3: stochastic volatility model with symmetric alpha-stable returns.
    * Both simulated & real data.
    * The paper itself is not Bayesian, the parameters are estimated through MLE.
    * Could implement using the PF and ABC and compare the results under model misspecification.

* [Nested particle filters for online parameter estimation in discrete-time state-space Markov models](https://projecteuclid.org/download/pdfview_1/euclid.bj/1522051233)
    * Section 6: three-dimensional Lorenz system static parameter estimation.
    * Some nasty stochastic differential equations, shown how to discretize.
    * The paper itself uses nested particle filters (? haven't read thoroughly).
    * The output variables are observed every 40 time steps, this has something to do with the nesting. We could simply take the model and observe at every time step as usual.

* [Approximate Bayesian Computation for a Class of Time Series Models](https://arxiv.org/pdf/1401.0265.pdf)
    * Again, stochastic volatility model using the S&P 500 index data.
    * Jasra.


* A lot of papers use simple linear Gaussian models. Could we include this & calculate optimally using the Kalman filter?
