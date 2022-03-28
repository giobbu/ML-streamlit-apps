# Mixtures of Gaussians and the EM algorithm

In the unsupervised learning setting we are given a training set  <img src="https://render.githubusercontent.com/render/math?math=\{x^{(1)}, ..., x^{(n)}\}"> without any labels. 

We wish to model the data by a joint distribution <img src="https://render.githubusercontent.com/render/math?math=p(x^{(i)}, z^{(i)})"> where <img src="https://render.githubusercontent.com/render/math?math=z^{(i)}">’s are latent (unobserved) random variables. 

### Hierarchy of steps

* <img src="https://render.githubusercontent.com/render/math?math=z(i) \sim Multinomial(\phi) ">
* <img src="https://render.githubusercontent.com/render/math?math=x(i) \middle|\ z(i)=j \sim \mathcal{N}(\mu_j,\sigma_j) ">



## Streamlit App
![Alt Text](./gmm.gif)

