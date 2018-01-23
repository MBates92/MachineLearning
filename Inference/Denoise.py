import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

#adds gaussian noise to an image
def add_gaussian_noise(im,prop,varSigma):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    e = varSigma*np.random.randn(np.prod(im.shape)).reshape(im.shape)
    im2 = np.copy(im)
    im2[index] += e[index]
    return im2

#adds binary noise to a binary image by flipping random pixel signs
def bin_noise(im,prop):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    im2 = np.copy(im)
    im2[index] = im2[index]*(-1)
    return im2

#adds salt and pepper noise to our image
def add_saltnpeppar_noise(im,prop):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    im2 = np.copy(im)
    im2[index] = 1-im2[index]
    return im2

#returns the neighbours of our pixel
def neighbours(i,j,M,N,size=4):
    if size==4:
        if (i==0 and j==0):
            n=[(0,1), (1,0)]
        elif i==0 and j==N-1:
            n=[(0,N-2), (1,N-1)]
        elif i==M-1 and j==0:
            n=[(M-1,1), (M-2,0)]
        elif i==M-1 and j==N-1:
            n=[(M-1,N-2), (M-2,N-1)]
        elif i==0:
            n=[(0,j-1), (0,j+1), (1,j)]
        elif i==M-1:
            n=[(M-1,j-1), (M-1,j+1), (M-2,j)]
        elif j==0:
            n=[(i-1,0), (i+1,0), (i,1)]
        elif j==N-1:
            n=[(i-1,N-1), (i+1,N-1), (i,N-2)]
        else:
            n=[(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
        return n
    if size==8:
        print('Not yet implemented\n')
        return -1

#Energy function of a pixel using n neighbours    
def energy_function(im,y,n,h,beta,nu,x_j):
        sum_1 = 0.
        sum_3 = 0.
        for k in range(0,len(n)):
            sum_1 += im[n[k]]
            sum_3 += im[n[k]]*y[n[k]]
        term_1 = sum_1*h
        term_2 = sum_1*-beta*x_j
        term_3 = -nu*sum_3
        return term_1+term_2+term_3
    
#Implement the ICM Ising model denoising    
def ICM(im,h,beta,nu):
    shape = np.shape(im)
    y = np.copy(im)
    M = shape[0]
    N = shape[1]
    for i in range(0,M):
        for j in range(0,N):
            n = neighbours(i,j,M,N)
            E_neg = energy_function(im,y,n,h,beta,nu,-1.0)
            E_pos = energy_function(im,y,n,h,beta,nu,1.0)
            if E_neg<E_pos:
                y[i,j] = 1.0
            else:
                y[i,j] = -1.0
    y_copy = np.copy(y)
    y_copy[np.where(y==-1.0)] = 1.0
    y_copy[np.where(y==1.0)] = -1.0
    return y_copy

#Implement the Gibbs Sampling Ising model denoising
def gibbs_ising(im,h,beta,nu):
    shape=np.shape(im)
    y = np.copy(im)
    M = shape[0]
    N = shape[1]
    for i in range(0,M):
        for j in range(0,N):
            n = neighbours(i,j,M,N)
            E_neg = energy_function(im,y,n,h,beta,nu,-1.0)
            E_pos = energy_function(im,y,n,h,beta,nu,1.0)
            posterior = E_pos/(E_pos+E_neg)
            t = np.random.random()
            if posterior>t:
                y[i,j] = 1.0
            else:
                y[i,j] = -1.0
    return y

#Gibbs Sampler Denoiser with random indexing
def gibbs_shuffle(im,h,beta,nu):
    shape=np.shape(im)
    y = np.copy(im)
    M = shape[0]
    N = shape[1]
    r = list(range(M))
    np.random.shuffle(r)
    for i in r:
        s = list(range(N))
        np.random.shuffle(s)
        for j in s:
            n = neighbours(i,j,M,N)
            E_neg = energy_function(im,y,n,h,beta,nu,-1.0)
            E_pos = energy_function(im,y,n,h,beta,nu,1.0)
            posterior = E_pos/(E_pos+E_neg)
            t = np.random.random()
            if posterior>t:
                y[i,j] = 1.0
            else:
                y[i,j] = -1.0
    return y

def mean_field(beta, mu, n):
    m = 0.0
    for k in range(0,len(n)):
        m += beta*mu[n[k]]
    return m

#Implement the Variational Bayes Ising denoising
def var_bayes(im,h,beta,nu,mu):
    shape=np.shape(im)
    y = np.copy(im)
    M = shape[0]
    N = shape[1]
    for i in range(0,M):
        for j in range(0,N):
            n = neighbours(i,j,M,N)
            E_neg = energy_function(im,y,n,h,beta,nu,-1.0)
            E_pos = energy_function(im,y,n,h,beta,nu,1.0)
            L_neg = (-1.0)*E_neg
            L_pos = (-1.0)*E_pos
            m = mean_field(beta, mu, n)
            posterior = sigmoid(2*(m + 0.5*(L_pos-L_neg)))
            mu[i,j] = np.tanh(m + 0.5*(L_pos-L_neg))
            if posterior >= 0.5:
                y[i,j] = 1.0
            else:
                y[i,j] = -1.0
            
    return y, mu

# proportion of pixels to alter
prop = 0.1
varSigma = 0.1
im = imread('cavapoo.png')
im = im/255

#Parameters for Energy function pulled from Bishop Ch. 8.3.3
h=0.0
beta = 3.0
nu = 1.0

#converting our image into a binary one
im_bin = np.copy(im)
im_bin[np.where(im_bin>=0.5)] = 1.
im_bin[np.where(im_bin<0.5)] = -1.

#Plotting our original binary image and adding noise
fig = plt.figure(figsize=(1920/144, 1080/144), dpi=144)
ax = fig.add_subplot(121)
ax.imshow(im_bin,cmap='gray')
ax.set_title('Original Binary Image')
im_bin_noise = bin_noise(im_bin,prop)
ax2 = fig.add_subplot(122)
ax2.imshow(im_bin_noise,cmap='gray')
ax2.set_title('Noisy Binary Image, '+str(prop*100)+' percent noise')
plt.savefig('Noisy_Image.png')

###############################################################################
###Implementing ICM denoising
###############################################################################
fig = plt.figure(figsize=(1920/144, 1080/144), dpi=144)
ax = fig.add_subplot(221)
ax.imshow(im_bin,cmap='gray')
ax.set_title('Original Binary Image')
ax2 = fig.add_subplot(222)
ax2.imshow(im_bin_noise,cmap='gray')
ax2.set_title('Noisy Binary Image, '+str(prop*100)+' percent noise')
ax3 = fig.add_subplot(223)
ax3.imshow(ICM(im_bin_noise,h,beta,nu),cmap='gray')
ax3.set_title('ICM Denoised Image, T=1')

#Implementing ICM denoising T times
T=5
denoised = np.copy(im_bin_noise)
for i in range(0,T):
    denoised = ICM(denoised,h,beta,nu)
ax4 = fig.add_subplot(224)
ax4.imshow(denoised,cmap='gray')
ax4.set_title('ICM Denoised Image, T='+str(T))
plt.savefig('ICM_Denoised_'+str(T)+'.png')

###############################################################################
###Implementing Gibbs Sampling Ising model
###############################################################################
fig = plt.figure(figsize=(1920/144, 1080/144), dpi=144)
ax = fig.add_subplot(221)
ax.imshow(im_bin,cmap='gray')
ax.set_title('Original Binary Image')
ax2 = fig.add_subplot(222)
ax2.imshow(im_bin_noise,cmap='gray')
ax2.set_title('Noisy Binary Image, '+str(prop*100)+' percent noise')
ax3 = fig.add_subplot(223)
ax3.imshow(gibbs_ising(im_bin_noise,h,beta,nu),cmap='gray')
ax3.set_title('Gibbs Index Ordered Denoised Image, T=1')

###Implementing Gibbs Sampling Ising model to denoise image, T times
T=5
gibbs_denoised = np.copy(im_bin_noise)
for i in range(0,T):
    gibbs_denoised = gibbs_ising(gibbs_denoised,h,beta,nu)
ax4 = fig.add_subplot(224)
ax4.imshow(gibbs_denoised,cmap='gray')
ax4.set_title('Gibbs Indexed Denoised Image, T='+str(T))
plt.savefig('Gibbs_Indexed_Denoised_0'+str(T)+'.png')

###############################################################################
###Implementing Gibbs Sampling Ising model shuffled indexes
###############################################################################
fig = plt.figure(figsize=(1920/144, 1080/144), dpi=144)
ax = fig.add_subplot(221)
ax.imshow(im_bin,cmap='gray')
ax.set_title('Original Binary Image')
ax2 = fig.add_subplot(222)
ax2.imshow(im_bin_noise,cmap='gray')
ax2.set_title('Noisy Binary Image, '+str(prop*100)+' percent noise')
ax3 = fig.add_subplot(223)
ax3.imshow(gibbs_shuffle(im_bin_noise,h,beta,nu),cmap='gray')
ax3.set_title('Gibbs Shuffle Denoised Image, T=1')

###Implementing Gibbs Sampling Ising model by shuffling, T times
T=5
gibbs = np.copy(im_bin_noise)
for i in range(0,T):
    gibbs = gibbs_shuffle(gibbs,h,beta,nu)
ax4 = fig.add_subplot(224)
ax4.imshow(gibbs,cmap='gray')
ax4.set_title('Gibbs Shuffle Denoised Image')
plt.savefig('Gibbs_Shuffle_Denoised_0'+str(T)+'.png')

###############################################################################
###Implementing the Variational Bayes method to the Noisy image.
###############################################################################
fig = plt.figure(figsize=(1920/144, 1080/144), dpi=144)
ax = fig.add_subplot(221)
ax.imshow(im_bin,cmap='gray')
ax.set_title('Original Binary Image')
ax2 = fig.add_subplot(222)
ax2.imshow(im_bin_noise,cmap='gray')
ax2.set_title('Noisy Binary Image, '+str(prop*100)+' percent noise')
mu = np.ones(im_bin_noise.shape)
ax3 = fig.add_subplot(223)
ax3.imshow(var_bayes(im_bin_noise,h,beta,nu,mu)[0],cmap='gray')
ax3.set_title('Variational Bayes Denoised Image, T=1')

###Implementing Variational Bayes method T times
T=5
var = np.copy(im_bin_noise)
shape = np.shape(var)
mu = np.ones(shape)
for i in range(0,T):
    var, mu = var_bayes(var,h,beta,nu,mu)
ax4 = fig.add_subplot(224)
ax4.imshow(var,cmap='gray')
ax4.set_title('Variational Bayes Denoised Image, T='+str(T))
plt.savefig('Variational_Bayes_Denoised_0'+str(T)+'.png')
