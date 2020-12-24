
from matplotlib import pyplot

import numpy

import numpy as np

def MUSIC_algo (recv_signal, D, M):

    # the wavelength of the wireless signal
    C = 3e8
    CHANNEL_FREQ = 2.442e9
    LAMBDA = C/CHANNEL_FREQ
    
    # antenna spacing: half lambda
    ANT_SPACING = LAMBDA/2
    

    # 1. compute the correlation matrix of the received signal.
    # the expected output matrix should be a [MxM] matrix
    # your code (5 point)
    # write three test cases to validate their correlation matrix is correct
    # check the photo for test case
    
    cor_matrix = np.matmul(recv_signal, recv_signal.conjugate().transpose())

    
    # 2. Compute the eigenvalues and the eigenvectors of the correlation matrix. 
    # You can use the numpy API to get the eigenvectors and eigenvalues: numpy.linalg.eig(input).
    # your code (5 point)
    
    eig_value, eig_vector = np.linalg.eigh(cor_matrix)

    # 3. sorting the eigenvalues and eigenvectors, extracting the eigenvectors correspond to the noises.
    # your code (10 point)
   

    
    sor_dic = {}
    for i in range(len(eig_value)):
        sor_dic[eig_value[i]]=eig_vector[:,i]

    import collections
    od = collections.OrderedDict(sorted(sor_dic.items()))

    noisy = []
    num = 0
    for key in od.keys():
        noisy.append(od[key])
        num = num + 1
        if num >= M-D:
            break
           
    noisy = np.asarray(noisy)

    # 4. Define the steering vector (referring to the defination of a(theta)). 
    # The spacing between adjacent antennas is LAMBDA/2.
    # your code (5 point)
    
    
    def steering(theta):
        a_theta = []
    
        for i in range(M):
            a_theta.append(np.exp(-1j*i*np.pi*np.sin(theta*np.pi/180)))
        a_theta = np.asarray(a_theta)
        return a_theta.reshape(a_theta.shape[0], 1)
    
    
    # 5. loop over the angle from -90 to 90 degrees in one degree increments, and compute the pseudo spectrum.
    # your code (5 point)
    
    p_spectrum = numpy.zeros(181)
    numm = 0
    for i in range(-90, 91):
        y = np.matmul(steering(i).transpose(), noisy.conjugate().transpose())
        z = np.linalg.norm(y)
        p_spectrum[numm] = 1.0/(z*z)
        numm = numm+1
    return p_spectrum


# #### Testing: (5 point)
# 
# Please call the following three testing cases and compare the result with the groundtruth give below.
#     

# In[15]:

# Testing case 1, arrival angle: [0.1 0.3 0.7]*pi;
import scipy.io



mat = scipy.io.loadmat('AoA_estimation_data_01.mat')
recv_signal = mat['csi_sample']
D = 3
M = 8
p_spectrum = MUSIC_algo(recv_signal,D,M)
pyplot.figure()
pyplot.title("Pseudospectrum Estimate via MUSIC algorithm")
pyplot.plot(np.linspace(-90, 90, 181), np.abs(p_spectrum), 'blue')
pyplot.show()


# Testing case 2, arrival angle: 0.3*pi, 0.7*pi
mat = scipy.io.loadmat('AoA_estimation_data_02.mat')
recv_signal = mat['csi_sample']
D = 2
M = 10
p_spectrum = MUSIC_algo(recv_signal,D,M)
pyplot.figure()
pyplot.title("Pseudospectrum Estimate via MUSIC algorithm")
pyplot.plot(np.linspace(-90, 90, 181), np.abs(p_spectrum), 'blue')
pyplot.show()
