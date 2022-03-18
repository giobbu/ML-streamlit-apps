from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
import random
import streamlit as st
from scipy.stats import multivariate_normal




st.title('Gaussian Mixture Model')



st.sidebar.markdown("## Find the number of cluster: ")

seed = st.sidebar.slider('SEED', 0, 1000, 1)

st.subheader('SEED: ' + str(seed))
np.random.seed(seed)
random.seed(seed)
n_samples = random.randint(500, 1000)
n = random.randint(2, 7)
interval = random.randint(1, 3)

st.sidebar.markdown("## GMM Hyper-parameters: ")
GMM_COMP = st.sidebar.slider( 'GMM COMPONENTS NUMBER', 1, 7, 1)
ITER = st.sidebar.selectbox( 'MAX ITER', (10, 20, 50, 100))


st.header('Dataset Generated')

list_gaussian = []

for i in range(0, n):

    C = np.array([[random.uniform(-interval, interval), random.uniform(-interval, interval)], [random.uniform(-interval, interval), random.uniform(-interval, interval)]])

    centre = random.randint(0, 10)
     
    gaussian = np.dot(np.random.randn(n_samples, 2) + np.array([centre, centre]) , C)

    list_gaussian.append(gaussian)

X_train = np.vstack(list_gaussian)

# plot samples
fig, ax = plt.subplots(figsize=(15, 10))
plt.scatter( X_train[:, 0], X_train[:, 1], 0.8)
st.pyplot(plt)

st.markdown("---")
st.header('GMM Fit')

# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components = GMM_COMP, covariance_type = "full", max_iter = ITER)
clf.fit(X_train)
BIC = clf.bic(X_train)
AIC = clf.aic(X_train)

col1, col2 = st.columns([1, 2])

with col1:
    st.text(f"BIC :  {BIC:.2f}")

with col2:
    st.text(f"AIC: {AIC:.2f}")

# display predicted scores by the model as a contour plot
x = np.linspace(-50.0, 50.0)
y = np.linspace(-50.0, 50.0)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

XX = np.array([X.ravel(), Y.ravel()]).T
Z = - clf.score_samples(XX)
Z = Z.reshape(X.shape)


if st.checkbox('Equi-probability surfaces'):
    CS = plt.contour( X, Y, Z, norm = LogNorm( vmin = 1.0, vmax = 500.0 ), levels = np.logspace(0, 5, 100)) # 



for comp in range(GMM_COMP):

    mu = clf.means_[comp]

    sd = clf.covariances_[comp]

    rv = multivariate_normal(clf.means_[comp], clf.covariances_[comp])

    neg_log = - rv.logpdf(pos)

    plt.contour(x, y, neg_log , norm = LogNorm( vmin = 1.0, vmax = 100.0 ), levels = np.logspace(0, 2, 10), cmap='RdGy')

    plt.scatter( mu[0], mu[1], color = "darkorange")

plt.suptitle("Negative log-likelihood predicted by a GMM", fontsize=20)
plt.axis("tight")
plt.show()
st.pyplot(plt)

st.markdown("---")
st.header(' Discover what the true number is ')
if st.checkbox('Click me'):
    
    st.subheader(f'The number of Gaussina distributions generating the data: - {n} -')
    # plot samples
    fig, ax = plt.subplots(figsize=(15, 10))
    for i in range(0, n):
            plt.scatter( X_train[ i * n_samples : (i+1) * n_samples , 0], X_train[ i * n_samples : (i+1) * n_samples, 1], 0.8)

    st.pyplot(plt)


