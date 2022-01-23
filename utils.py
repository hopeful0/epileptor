import numpy as np
import scipy.io as scio
from ctypes import *
from scipy.signal import butter, lfilter, lfilter_zi
from scipy.signal import hilbert

# convert m*n 2d c pointer into a np array
def cp2nparry(cp, m, n):
    arr = np.zeros((m, n))
    for i in range(m):
        arr[i] = cp[i][:n]
    return arr

# convert m*n 2d nparray into a 2d c pointer 
def array2cp(arr, m, n, dtype=c_double):
    cp = (POINTER(dtype) * m)()
    for i in range(m):
        cp[i] = (dtype * n)()
        for j in range(n):
            cp[i][j] = arr[i][j]
    return cp

def np2carray(arr, n):
    carray = (c_double * n)()
    for i in range(n):
        carray[i] = arr[i]
        
    return carray

# save the epileptor result to a mat file
# m is the network size and n is the number of iterations
def saveResult2Mat(result, filename, m, n):
    scio.savemat(filename, {
        'x_1s': cp2nparry(result.x_1s, m, n),
        'zs': cp2nparry(result.zs, m, n),
        'x_2s': cp2nparry(result.x_2s, m, n),
        'Izs': cp2nparry(result.Izs, m, n),
    })

# load structural connection
def loadConnection():
    connection =  scio.loadmat('./data/conn_aal.mat')['conn']
    connection = connection / np.max(connection)
    return connection

def loadDistance():
    distance = np.random.randint(1, 111, size=(116, 116))
    for i in range(116):
        distance[i, i] = 0
    return distance

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y, _ = lfilter(b, a, data, zi=zi*data[0])
    return y

# Phase Synchronization
def phase_sync(x, y):
    hx = hilbert(x)
    hy = hilbert(y)
    px = np.unwrap(np.angle(hx))
    py = np.unwrap(np.angle(hy))
    return np.abs(np.sum(np.exp((px - py) * 1j))) / np.size(px)

def extract_ts(result, length, i, offset=0, filtered=False):
    ts = result.x_1s[i][offset:offset + length]
    for ii in range(length):
        ts[ii] += result.x_2s[i][offset + ii]
    if filtered:
        ts = butter_bandpass_filter(ts, 0.16, 97, 256)
    return ts

# def ses2propagations(ses):
#     pass