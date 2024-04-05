import os
import numpy as np  # scientific computing module with Python
import numba as nb  # high performance Python compiler
import time as time 


# simulation parameters
M = eval(os.environ.get('M', '1000'))  # number of partition point of time t
N = eval(os.environ.get('N', '10'))  # number of partition point of state y
L = eval(os.environ.get('L', '1000000'))  # number of simulation times

delta_t = eval(os.environ.get('DELTA_T', '0.001'))
delta_y = eval(os.environ.get('DELTA_Y', '0.01'))
t_0 = 0  # start time point
t_M = eval(os.environ.get('T_M', '1'))  # end time point T
y_0 = eval(os.environ.get('Y_0', '0.1'))  # minimal value of initial state y
y_N = eval(os.environ.get('Y_N', '0.2'))  # maximal value of initial state y

assert np.allclose(t_M, t_0 + delta_t * M)
assert np.allclose(y_N, y_0 + delta_y * N)

# CEVSV model parameters
v = eval(os.environ.get('V', '0.5'))  # 0.5 is Heston's model
r = eval(os.environ.get('R', '0.04'))
kappa = eval(os.environ.get('KAPPA', '1.5'))
theta = eval(os.environ.get('THETA', '0.2'))
sigmav = eval(os.environ.get('SIGMAV', '0.25'))
lambda_ = eval(os.environ.get('LAMBDA_', '0.5'))
rho = eval(os.environ.get('RHO', '-0.8'))
gamma = eval(os.environ.get('GAMMA', '2'))

### line 5 to line 28: set environment variables 

print(
    f'Parameter Sets: \
v = {v}; r = {r}; kappa = {kappa}; \
theta = {theta}; sigmav = {sigmav}; \
lambda_ = {lambda_}; rho = {rho}; gamma = {gamma}'
)

# global constant expressions
sqrt_delta_t = np.sqrt(delta_t)
sqrt_1subrho2 = np.sqrt(1 - rho ** 2)
lambda2 = lambda_ ** 2
vsub1 = v - 1

gamma_1 = (1 - gamma) / gamma
gamma_2 = (1 - gamma) / (2 * gamma)

# inline small functions to speed up

@nb.njit(inline='always')
def next_Ys(Yt, zmix):
    """
    Move state $Y_t$ to the next time point.

    :param Yt: initial value of state, shape L*1
    :param zmix: composite standard normal value, shape L*1
        that is used to approximate Brownian motion
    :returns: $Y_{t+1}$ shape L*1
    """
    # truncate extremely large y to aviod explosion
    ans = Yt + kappa * (theta - Yt) * delta_t + sigmav * Yt**v * sqrt_delta_t * zmix
    # To make sure each element is no larger than 4 
    ans_1 = -1*np.abs(ans-4)/2 + (ans+4)/2
    #To make sure that the volatility is positive
    ans_1 = 0.5*ans+np.abs(ans_1)*0.5
    return ans_1

@nb.njit(inline='always')
def next_DYs(DYt, Yt, zmix):
    """
    Move Mallivin derivative $DY_t$ to the next time point.

    :param DYt: initial value of Mallivin derivative,shape L*1
    :param Yt: initial value of state,shape L*1
    :param zmix: composite standard normal value
        that is used to approximate Brownian motion
    :returns: $DY_{t+1}$
    """
    return DYt * (1 - kappa * delta_t + sigmav * v * Yt**vsub1 * sqrt_delta_t * zmix)


@nb.njit(inline='always')
def get_thetah(Yt):
    r"""
    Evaluate $\theta^h$ given current state.

    :param Yt: current state, shape L*1
    :returns: thetah at current state
    """
    return lambda_ * np.sqrt(Yt)


@nb.njit(inline='always')
def next_logxih(thetah, z1):
    r"""
    Evaluate $log(\xi^h)$ given current $\theta^h$ and simulated standard normal value.

    :param thetah: current $\theta^h$,shape L*1
    :param z1: simulated standard normal value
        that is used to approximate Brownian motion $W_1$,shape L*1
    :returns: $log(\xi^h)$ at current time interval $\delta t$
    """
    return gamma_2 * thetah**2 * delta_t + gamma_1 * thetah * sqrt_delta_t * z1


@nb.njit(inline='always')
def next_logxiu(thetau, z2):
    r"""
    Evaluate $log(\xi^h)$ given current $\theta^u$ and simulated standard normal value.

    :param thetah: current $\theta^u$
    :param z2: simulated standard normal value
        that is used to approximate Brownian motion $W_2$
    :returns: $log(\xi^u)$ at current time interval $\delta t$
    """
    return gamma_2 * thetau**2 * delta_t + gamma_1 * thetau * sqrt_delta_t * z2


@nb.njit(inline='always')
def next_Hthetah(thetah, DYs, z1):
    r"""
    Evaluate $log(H_\theta^h)$ given current $\theta^h$ and simulated standard normal value.

    :param thetah: current $\theta^h$ shape L*1
    :param DYs: current $DY_t$ shape L*1
    :param z1: simulated standard normal value shape L*1
        that is used to approximate Brownian motion $W_1$
    :returns: $log(H_\theta^h)$ at current time interval $\delta t$
    """
    return lambda2 / 2 * DYs * (delta_t + sqrt_delta_t * z1 / thetah)


@nb.njit(inline='always')
def next_Hthetau(thetau, thetauy, DYs, z2):
    r"""
    Evaluate $log(H_\theta^u)$ given current $\theta^u$ and simulated standard normal value.

    :param thetah: current $\theta^u$
    :param thetauy: current $\theta^u_y$
    :param DYs: current $DY_t$
    :shape L*1
    :param z2: simulated standard normal value
        that is used to approximate Brownian motion $W_2$
    :returns: $log(H_\theta^u)$ at current time interval $\delta t$
    """
    return DYs * (thetau * thetauy * delta_t + thetauy * sqrt_delta_t * z2)


@nb.njit(inline='always')
def get_thetau(   ##interpolation to get thetau
    theta_ui, theta_uyi, Yt,
):
    """
    Given value of thetau and thetauy at discrete points, return value of thetau
    and thetauy at current state Yt through interpolation method.

    :param theta_ui: an array with shape (N+1,) storing thetau value at each state
    :shape 1*N
    :param theta_uyi: an array with shape (N+1,) storing thetauy value at each state
    :shape 1*N
    :param Yt: current state 
    :shape L*1
    :returns: value of thetau and thetauy at this current state
    """
    L = len(Yt)
    ans_1 = np.full(Yt.shape, 0)
    ans_2 = np.full(Yt.shape, 0)
    def find(y):
        if y < y_0:  # interpolation with first and second value
            theta_ut = (theta_ui[1] - theta_ui[0]) / delta_y * (y - y_0) + theta_ui[0]
            theta_uyt = (theta_uyi[1] - theta_uyi[0]) / delta_y * (y - y_0) + theta_uyi[0]
        elif y > y_N:  # interpolation with last and second last value
            theta_ut = (theta_ui[-1] - theta_ui[-2]) / delta_y * (y - y_N) + theta_ui[-1]
            theta_uyt = (theta_uyi[-1] - theta_uyi[-2]) / delta_y * (y - y_N) + theta_uyi[-1]
        else:  # y_0 <= yij <= y_N, linear interpolation
    
            tmp = (y - y_0) / delta_y
            yleft = int(tmp)
    
            # bound check
            if yleft != N:
                wright = tmp - yleft
                theta_ut = theta_ui[yleft] * (1 - wright) + theta_ui[yleft+1] * wright
                theta_uyt = theta_uyi[yleft] * (1 - wright) + theta_uyi[yleft+1] * wright
            else:
                theta_ut = theta_ui[yleft]
                theta_uyt = theta_uyi[yleft]
        return theta_ut, theta_uyt
    
    temp = np.array(list(map(find,Yt)))
    return temp[:,0],temp[:,1]

    # '''
    # Case 1: Yt<y_0
    # '''
    
    # index_1 = np.where(Yt<y_0)[0]
    # val = Yt[index_1]
    # temp_1 = (theta_ui[1] - theta_ui[0]) / delta_y * (val - y_0) + theta_ui[0]
    # temp_2 = (theta_uyi[1] - theta_uyi[0]) / delta_y * (val - y_0) + theta_uyi[0]
    # ans_1[index_1] = temp_1.copy()
    # ans_2[index_1] = temp_2.copy()
    # '''
    # Case 2: Yt>y_N
    # '''
    # index_2 = np.where(Yt>=y_N)[0]
    # val = Yt[index_2]
    # temp_3 = (theta_ui[-1] - theta_ui[-2]) / delta_y * (val - y_N) + theta_ui[-1]
    # temp_4 = (theta_uyi[-1] - theta_uyi[-2]) / delta_y * (val - y_N) + theta_uyi[-1]
    # ans_1[index_2] = temp_3
    # ans_2[index_2] = temp_4    
    # '''
    # Case 3: Yt<y_0
    # '''
    # index_3 = np.where((Yt>y_0)&(Yt<y_N))[0]
    # val = Yt[index_3]
    # tmp = (val-y_0)/delta_y
    # yleft = tmp.astype(int)
    # wleft = tmp-yleft
    # yright = yleft + 1 
    
    # theta_u_tile = np.tile(theta_ui,(L,1))
    # theta_uy_tile = np.tile(theta_uyi,(L,1))
    # temp_5 = wleft*theta_u_tile[:,yleft]+(1-wleft)*theta_u_tile[:,yright+1]
    # temp_6 = wleft*theta_uy_tile[:,yleft]+(1-wleft)*theta_uy_tile[:,yright+1]
    # ans_1[index_3] = temp_5
    # ans_2[index_3] = temp_6
    
    return ans_1, ans_2
    
    

@nb.njit(inline='always')
def get_theta_uy(theta_ui):
    """
    Calculate numerical derivative with respect to y at time ti for all yj.
    The shape of theta_ui should be (m,) that satisfies m >= 3.
    shape of the theta_ui 1*N at the time t 

    :param theta_ui: an array with shape (N+1,) storing thetau value at each state
    :returns theta_uyi: an array with shape (N+1,) storing thetauy value at each state
    """
    m = len(theta_ui)
    theta_uyi = np.empty((m,), dtype=theta_ui.dtype)

    two_delta_y = 2 * delta_y

    for i in range(1, m-1):
        # three mid point formula
        theta_uyi[i] = (theta_ui[i+1] - theta_ui[i-1]) / two_delta_y

    # three end point formula
    theta_uyi[0] = (4*theta_ui[1] - 3*theta_ui[0] - theta_ui[2]) / two_delta_y
    theta_uyi[-1] = (3*theta_ui[-1] - 4*theta_ui[-2] + theta_ui[-3]) / two_delta_y
    return theta_uyi


# @nb.njit(   ## main part 
#     parallel=True,  # parallel iteration based on OpenMP
#     fastmath=True,  # enable LLVM fastmath to speed up
#     nogil=True  # release Python global interpreter lock (GIL) to speed up
# )
def simu_thetau(theta_u, theta_uy, z1, z2):  
    """
    Simulate theta_u.

    :param theta_u: an array with shape (M+1, N+1) storing simulated value of thetau
    :param theta_uy: an array with shape (M+1, N+1) storing estimation value of thetauy
    :param z1: an array with shape (L, M+1) storing standard normal values
        that is used to approximate Brownian motion $W_1$
    :param z2: an array with shape (L, M+1) storing standard normal values
        that is used to approximate Brownian motion $W_2$
    :returns xi, xiHtheta: xi is the denominator value for each simulation path at the
        final step. xiHtheta is the numerator value for each simulation path at the
        final step. They are used to estimate the standard error of this simulation.
    """
    # zmix is the composite standard normal value that
    # will be used to simulate state variable
    zmix = (rho * z1 + sqrt_1subrho2 * z2)  # (L, M+1)
    L, M = zmix.shape[0], zmix.shape[1] - 1  # make the variable local to make numba happy

    # initial value
    Y_0 = np.arange(0, N+1) * delta_y + y_0
    DY_0 = sigmav * (Y_0**v) * sqrt_1subrho2

    # Temp value of $log(\xi^h)$ and $H_{\theta}^h$ These values do not depend on theta_u
    # and can be reused across time. for each time step, adding the increment part
    # is enough and there is no need to evaluate through the entire path again.
    # However, for $log(\xi^u) and $H_{\theta}^u$, which depend on theta_u,
    # we must relies on the full path to evaluate them.
    logxih = np.zeros((L, N+1))
    Hthetah = np.zeros((L, N+1))

    # For each simulation path l (0 <= l < L),
    # store the simulated numerator value (xiHtheta)
    # and the simulated denominator value (xi) of thetau.
    xi = np.empty((L, N+1))
    xiHtheta = np.empty((L, N+1))

    # thetau is solved backwardly
    for i in range(M-1, -1, -1):   ## backward simulation 
        # m is number of forward simulation steps
        m = M-i

        # future value of thetau and thetauy
        # Notice that theta_u[i] is what we want to derive now.
        # Since we use left point to evaluate integrals and
        # theta_u[i] is currently unknown,
        # my algorithm will simply drop the integral value on the
        # first small interval, which is equivalent to use
        # zero as theta_u's initial value
        theta_ui = theta_u[i:]
        theta_uyi = theta_uy[i:]
        zmix_i = zmix[:, 1:m+1]
        z1_i = z1[:, 1:m+1]
        z2_i = z2[:, 1:m+1]
        for j in range(0,N+1):
            Y_k = np.ones((L))*Y_0[j]
            DY_k = np.ones((L))*DY_0[j]
            logxiu = np.zeros((L))
            Hthetau = np.zeros((L))
            for k in range(m-1):
                #this function has problem 
                
                
                
                a_1= theta_ui[k]
                b_1 = theta_uyi[k]
                Yt = Y_k
                L = len(Yt)
    
                def find(a,b,y):
                    if y < y_0:  # interpolation with first and second value
                        theta_ut = (a[1] - a[0]) / delta_y * (y - y_0) + a[0]
                        theta_uyt = (b[1] - b[0]) / delta_y * (y - y_0) + b[0]
                    elif y > y_N:  # interpolation with last and second last value
                        theta_ut = (a[-1] - a[-2]) / delta_y * (y - y_N) + a[-1]
                        theta_uyt = (b[-1] - b[-2]) / delta_y * (y - y_N) + b[-1]
                    else:  # y_0 <= yij <= y_N, linear interpolation
                
                        tmp = (y - y_0) / delta_y
                        yleft = int(tmp)
                
                        # bound check
                        if yleft != N:
                            wright = tmp - yleft
                            theta_ut = a[yleft] * (1 - wright) + a[yleft+1] * wright
                            theta_uyt = b[yleft] * (1 - wright) + b[yleft+1] * wright
                        else:
                            theta_ut = a[yleft]
                            theta_uyt = b[yleft]
                    return theta_ut, theta_uyt
                a_input = np.vstack([a_1] * L)
                b_input = np.vstack([b_1] * L)
                
                temp = np.array(list(map(find,a_input,b_input,Yt)))
                theta_u_k, theta_uy_k = temp[:,0],temp[:,1]
                
                
                
                
                
                
                
                
             
                logxiu += next_logxiu(theta_u_k, z2_i[:,k])
                Hthetau += next_Hthetau(theta_u_k, theta_uy_k, DY_k, z2_i[:,k])
                
                # move state variable Y and Mallivin derivative DY to the next time point
                DY_k = next_DYs(DY_k, Y_k, zmix_i[:,k])
                #this function has problem 
                Y_k = next_Ys(Y_k, zmix_i[:,k])
            
            theta_h_k = get_thetah(Y_k)
            logxih[:, j] +=  next_logxih(theta_h_k, z1_i[:,-1])
            Hthetah[:, j] +=  next_Hthetah(theta_h_k, DY_k, z1_i[:,-1])
            
            xi_tmp = np.exp(logxih[:, j] + logxiu)
            Htheta_tmp = Hthetah[:, j] + Hthetau

            # store simualted numerator value and denominator value of thetau
            xi[:, j] =  xi_tmp
            xiHtheta[:, j] = xi_tmp * Htheta_tmp
        
        xi_sum = xi.sum(axis=0)
        xiHtheta_sum = xiHtheta.sum(axis=0)
        theta_u[i] = (gamma - 1.) * xiHtheta_sum / xi_sum
        theta_uy[i] = get_theta_uy(theta_u[i])

        # output thetau value to show the progress
        if i % 10 == 0:
            print(i, N // 4, theta_u[i, N // 4])
        
        
        # the simualted value of each path at the final time step
        # is returned in order to calculate standard error
        return xi, xiHtheta
                
            

  
if __name__ == '__main__':

    # set a random seed
    np.random.seed(seed=50)
    t1 = time.time()
    # initialize thetau and its first derivative with respect to y
    theta_u = np.zeros((M+1, N+1), np.float64)
    theta_uy = np.zeros((M+1, N+1), np.float64)

    # independent standard normal values
    z1 = np.random.standard_normal(size=(L, M+1))
    z2 = np.random.standard_normal(size=(L, M+1))
    t2 = time.time()
    print("Generate all the random variables")
    # start the simualtion
    xi, xiHtheta = simu_thetau(theta_u, theta_uy, z1, z2)
    t3 = time.time()
    print("Finishing the iteration")
    print("{} seconds to generate the r.v".format(t2-t1))
    print("{} seconds for simulation".format(t3 - t2))
    

    # save result
    np.savetxt('theta_u_try.csv', theta_u, delimiter=',')
    np.savetxt('xi{_try.csv', xi, delimiter=',')
    np.savetxt('xiHtheta_try.csv', xiHtheta, delimiter=',')

    

