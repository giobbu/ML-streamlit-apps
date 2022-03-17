from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
import random
import streamlit as st
from scipy.stats import multivariate_normal


@st.cache
def generate_random_number():
    return random.randint(1,1000)


st.title('Gaussian Mixture Model')

seed = generate_random_number()
st.write('SEED: ' + str(seed))
# generate random sample, two components
np.random.seed(seed)
random.seed(seed)

n_samples = st.selectbox( 'NUMBER SAMPLES', (100, 500, 1000))
gmm_components = st.selectbox( 'GMM NUMBER COMPONENTS', (2, 5, 10))
ITER = st.selectbox( 'MAX ITER', (10, 20, 50, 100))

n = 5 # HIDDEN NUMBER
interval = random.randint(1, 3)


list_gaussian = []

for i in range(1, n):

    C = np.array([[random.uniform(-interval, interval), random.uniform(-interval, interval)], [random.uniform(-interval, interval), random.uniform(-interval, interval)]])

    centre = random.randint(0, 10)
     
    stretched_gaussian = np.dot(np.random.randn(n_samples, 2) + np.array([centre, centre]) , C)

    list_gaussian.append(stretched_gaussian)

X_train = np.vstack(list_gaussian)


# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components = gmm_components, covariance_type = "full", max_iter = ITER)
clf.fit(X_train)

BIC = clf.bic(X_train)
AIC = clf.aic(X_train)
st.write(f"BIC :  {BIC:.2f}          AIC: {AIC:.2f}")


# display predicted scores by the model as a contour plot
x = np.linspace(-50.0, 50.0)
y = np.linspace(-50.0, 50.0)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

XX = np.array([X.ravel(), Y.ravel()]).T
Z = - clf.score_samples(XX)
Z = Z.reshape(X.shape)

fig, ax = plt.subplots(figsize=(15, 10))

CS = plt.contour( X, Y, Z, norm = LogNorm( vmin = 1.0, vmax = 500.0 ), levels = np.logspace(0, 3, 100)) # 

# plot samples
plt.scatter( X_train[:, 0], X_train[:, 1], 0.8)

for comp in range(gmm_components):

    mu = clf.means_[comp]

    sd = clf.covariances_[comp]

    rv = multivariate_normal(clf.means_[comp], clf.covariances_[comp])

    neg_log = - rv.logpdf(pos)

    plt.contour(x, y, neg_log , norm = LogNorm( vmin = 1.0, vmax = 100.0 ), levels = np.logspace(0, 3, 10), cmap='RdGy')

    plt.scatter( mu[0], mu[1], color = "darkorange")

plt.title("Negative log-likelihood predicted by a GMM")
plt.axis("tight")
plt.show()



st.pyplot(plt)

