# Stochastic optimizers in Deep Learning

-  'SGD-G2' : pytorch implementation of the second-order stochastic optimizer based on Runge Kutta scheme. This optimizer is introduced in our paper **I. Ayadi and G. Turinici. “Stochastic Runge-Kutta methods and adaptive SGD-G2 stochastic
gradient descent.” 2020 25th International Conference on Pattern Recognition (ICPR). IEEE, 2021. doi:https://doi.org/10.1109/ICPR48806.2021.9412831**

- 'ADAM' : pytorch re-implementation of the well-known ADAM optimizer (with a slight change in the update of weights consisting in replacing "\sqrt{\hat{v}_t}+\epsilon"with "\sqrt{\hat{v}_t+\epsilon^2}"

- 'PseudoSGD' : pytorch implementation of a SGD-like optimizer that approximates ADAM in the learning rate with order 2 (in the weak sense)
