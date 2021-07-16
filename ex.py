import numba
from numba import vectorize
from numba.pycc import CC
import numpy as np
import math
import matplotlib.pyplot as plt
import time

np.random.seed(1234567890)

cc = CC('my_module')
# Uncomment the following line to print out the compilation steps
cc.verbose = True

#@cc.export('log_gaussian_numba_cc', 'f8(f8[:],f8, f8)')
#def log_gaussian_numba_cc(x, mean, sigma):
#    log_pdf = -(x - mean) ** 2 / (2 * sigma ** 2)
#    log_pdf = log_pdf - np.log((np.sqrt(2 * np.pi) * sigma))
#    return log_pdf


def plot(x,y):
    # using the variable axs for multiple Axes
    fig, axs = plt.subplots( nrows=2, ncols=1,
                             sharex=False, sharey=False,
                             squeeze=True, subplot_kw=None,
                             gridspec_kw=None,
                             figsize=(10,12))
    #plt.tight_layout()

    # 
    axs[0].plot(x, y)
    #axs[0].set_xbound(-0.5,5.5)
    #axs[0].set_ybound(-0.5,5.5)
    #axs[0].set_title('The title 0')
    #axs[0].set_xlabel('The x lable 0')
    #axs[0].set_ylabel('The y lable 0')
    axs[0].set_position([0.05,0.55,0.9,0.40])
    #
    axs[1].scatter(x, y)
    #axs[1].set_xbound(-0.5,5.5)
    #axs[1].set_ybound(-0.5,5.5)
    #axs[1].set_title('The title 1')
    #axs[1].set_xlabel('The x lable 1')
    #axs[1].set_ylabel('The y lable 1')
    axs[1].set_position([0.05,0.05,0.9,0.40])

    plt.show()

#@numba.njit(nopython=True)
@numba.jit(nopython=True)
def log_gaussian_numba(x, mean, sigma):
    log_pdf = -(x - mean) ** 2 / (2 * sigma ** 2)
    log_pdf = log_pdf - np.log((np.sqrt(2 * np.pi) * sigma))
    return log_pdf

def log_gaussian(x, mean, sigma):
    log_pdf = -(x - mean) ** 2 / (2 * sigma ** 2)
    log_pdf = log_pdf - np.log((np.sqrt(2 * np.pi) * sigma))
    return log_pdf

@vectorize
def log_gaussian_numba_vectorize(x, mean, sigma):
    log_pdf = -(x - mean) ** 2 / (2 * sigma ** 2)
    log_pdf = log_pdf - np.log((np.sqrt(2 * np.pi) * sigma))
    return log_pdf

@numba.jit(nopython=True)
def gogo_numba(function, x, mean, sigma):
    for i in range(nn):
        y = function(x, mean, sigma)

def gogo(function, x, mean, sigma):
    for i in range(nn):
        y = function(x, mean, sigma)

def main():

    nPoints = 1000
    xMin = -150
    xMax =  150

    mean = 0.0
    sigma = 1.0

    log_gaussian_v = np.vectorize(log_gaussian)
    #log_gaussian_numba_v = np.vectorize(log_gaussian_numba)

    x = np.linspace(xMin, xMax, nPoints)
    
    time01 = time.time()
    gogo(log_gaussian, x, mean, sigma)
    time02 = time.time()
    print('PYTHON  ',time02-time01)
    
    #time01 = time.time()
    #gogo(log_gaussian_numba_vectorize, x, mean, sigma)
    #time02 = time.time()
    #print('NUMBA vectorize ',time02-time01)

    #time01 = time.time()
    #gogo(log_gaussian_v, x, mean, sigma)
    #time02 = time.time()
    #print('NumPy vectorize ',time02-time01)

    time01 = time.time()
    gogo(log_gaussian_numba, x, mean, sigma)
    time02 = time.time()
    print('NUMBA   ',time02-time01)

    time01 = time.time()
    gogo_numba(log_gaussian_numba, x, mean, sigma)
    time02 = time.time()
    print('NUMBA2  ',time02-time01)

    time01 = time.time()
    gogo_numba(log_gaussian_numba, x, mean, sigma)
    time02 = time.time()
    print('NUMBA3  ',time02-time01)

    time01 = time.time()
    gogo_numba(log_gaussian_numba, x, mean, sigma)
    time02 = time.time()
    print('NUMBA4  ',time02-time01)
    
    '''
    try:
        time01 = time.time()
        gogo_numba(log_gaussian_v, x, mean, sigma)
        time02 = time.time()
        print('YES NO      ',time02-time01)
    except:
        print('YES NO      ','ERROR')
    '''
    
    '''
    try:
        time01 = time.time()
        gogo_numba(log_gaussian_numba_vectorize, x, mean, sigma)
        time02 = time.time()
        print('YES YES     ',time02-time01)
    except:
        print('YES YES     ','ERROR')
    '''


@cc.export('multf', 'f8(f8, f8)')
@cc.export('multi', 'i4(i4, i4)')
def mult(a, b):
    return a * b

@cc.export('square', 'f8(f8)')
def square(a):
    return a ** 2


#import my_module
#print(my_module.multi(3, 4))
#print(my_module.square(1.414))

nn = 7000

if __name__ == "__main__":
    main()
    cc.compile()
