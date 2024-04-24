import numpy as np 
import numba as nb 
def simu_thetau(theta_u, theta_uy, z1, z2):  
    zmix = (0.2 * z1 + 0.1 * z2)  
    L, M = zmix.shape[0], zmix.shape[1] - 1  
    N = 100
    Y_0 = np.arange(0, N+1) * 0.1 + 0
    logxih = np.zeros((L, N+1))
    Hthetah = np.zeros((L, N+1))
    xi = np.empty((L, N+1))
    xiHtheta = np.empty((L, N+1))
    for i in range(M-1, -1, -1):  
        m = M-i
        for l in nb.prange(L):
            zmix_i = zmix[l, 1:m+1]
            for j in range(0, N+1):
                Y_k = Y_0[j]
                for k in range(m-1):
                    Y_k = zmix_i[k]
                theta_h_k = (Y_k)
                
        
      

