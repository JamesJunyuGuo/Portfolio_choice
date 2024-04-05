import os
import numpy as np  # scientific computing module with Python
import numba as nb  # high performance Python compiler

# simulation parameters
M = eval(os.environ.get('M', '200'))  # number of partition point of time t
N = eval(os.environ.get('N', '10'))  # number of partition point of state y
L = eval(os.environ.get('L', '10000'))  # number of simulation times
lst = np.zeros((M,2,L,N))

delta_y = eval(os.environ.get('DELTA_Y', '0.01'))
t_0 = 0  # start time point
t_M = eval(os.environ.get('T_M', '1'))  # end time point T
y_0 = eval(os.environ.get('Y_0', '0.1'))  # minimal value of initial state y
y_N = eval(os.environ.get('Y_N', '0.2'))  # maximal value of initial state y
delta_t = eval(os.environ.get('DELTA_T', '0.001'))
delta_t = (t_M-t_0)/M 
delta_y = (y_N-y_0)/N

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

    :param Yt: initial value of state
    :param zmix: composite standard normal value
        that is used to approximate Brownian motion
    :returns: $Y_{t+1}$
    """
    # truncate extremely large y to aviod explosion
    return min(Yt + kappa * (theta - Yt) * delta_t + sigmav * Yt**v * sqrt_delta_t * zmix, 4)


@nb.njit(inline='always')
def next_DYs(DYt, Yt, zmix):
    """
    Move Mallivin derivative $DY_t$ to the next time point.

    :param DYt: initial value of Mallivin derivative
    :param Yt: initial value of state
    :param zmix: composite standard normal value
        that is used to approximate Brownian motion
    :returns: $DY_{t+1}$
    """
    return DYt * (1 - kappa * delta_t + sigmav * v * Yt**vsub1 * sqrt_delta_t * zmix)


@nb.njit(inline='always')
def get_thetah(Yt):
    r"""
    Evaluate $\theta^h$ given current state.

    :param Yt: current state
    :returns: thetah at current state
    """
    return lambda_ * np.sqrt(Yt)


@nb.njit(inline='always')
def next_logxih(thetah, z1):
    r"""
    Evaluate $log(\xi^h)$ given current $\theta^h$ and simulated standard normal value.

    :param thetah: current $\theta^h$
    :param z1: simulated standard normal value
        that is used to approximate Brownian motion $W_1$
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

    :param thetah: current $\theta^h$
    :param DYs: current $DY_t$
    :param z1: simulated standard normal value
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
    :param theta_uyi: an array with shape (N+1,) storing thetauy value at each state
    :param Yt: current state
    :returns: value of thetau and thetauy at this current state
    """
    if Yt < y_0:  # interpolation with first and second value
        theta_ut = (theta_ui[1] - theta_ui[0]) / delta_y * (Yt - y_0) + theta_ui[0]
        theta_uyt = (theta_uyi[1] - theta_uyi[0]) / delta_y * (Yt - y_0) + theta_uyi[0]
    elif Yt > y_N:  # interpolation with last and second last value
        theta_ut = (theta_ui[-1] - theta_ui[-2]) / delta_y * (Yt - y_N) + theta_ui[-1]
        theta_uyt = (theta_uyi[-1] - theta_uyi[-2]) / delta_y * (Yt - y_N) + theta_uyi[-1]
    else:  # y_0 <= yij <= y_N, linear interpolation

        tmp = (Yt - y_0) / delta_y
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


@nb.jit(inline='always')
def discretize(func,Yt):
    if Yt < y_0:  # interpolation with first and second value
        func_ans  = (func[1] - func[0]) / delta_y * (Yt - y_0) + func[0]

    elif Yt > y_N:  # interpolation with last and second last value
        func_ans = (func[-1] - func[-2]) / delta_y * (Yt - y_N) + func[-1]

    else:  # y_0 <= yij <= y_N, linear interpolation

        tmp = (Yt - y_0) / delta_y
        yleft = int(tmp)
        # bound check
        if yleft != N:
            wright = tmp - yleft
            func_ans = func[yleft] * (1 - wright) + func[yleft+1] * wright

        else:
            func_ans = func[yleft]
    return func_ans
    


@nb.njit(inline='always')
def get_theta_uy(theta_ui):
    """
    Calculate numerical derivative with respect to y at time ti for all yj.
    The shape of theta_ui should be (m,) that satisfies m >= 3.

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


    


@nb.njit(inline='always')
def sim_Ht1t2(DYs,Ys,theta_u1,theta_uy1,z1,z2):
    '''
    Use the input to simulate the H*xi from t_i to t_{i+1} using the Ito formula 
    : DYs : The Malliavin derivative
    : Ys : the current state variable 
    : theta_u1: we use the theta value on the next grid point 
    : theta_uy1: we use the theta_uy value on the next grid poin t
    : z1,z2 represent the brownian motion
    '''
    theta_h1 = get_thetah(Ys)
    drift = DYs*(lambda2/2+theta_u1*theta_uy1)*delta_t +((1-gamma)/gamma)*DYs*(lambda2/2+theta_u1*theta_uy1)*delta_t 
    diffusion  = DYs*((lambda2/2/theta_h1) * z1*sqrt_delta_t + theta_uy1*z2*sqrt_delta_t)
    return drift + diffusion


@nb.njit(   ## main part 
    parallel=True,  # parallel iteration based on OpenMP
    fastmath=True,  # enable LLVM fastmath to speed up
    nogil=True  # release Python global interpreter lock (GIL) to speed up
)
def simu_thetau(theta_u, theta_uy, z1, z2,lst,lst_1):  
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
    # Hthetah = np.zeros((L, N+1))

    # For each simulation path l (0 <= l < L),
    # store the simulated numerator value (xiHtheta)
    # and the simulated denominator value (xi) of thetau.
    xi = np.empty((L, N+1))
    xiHtheta = np.empty((L, N+1))
    
    xi_sum = np.ones((N+1))
    xiHtheta_sum = np.zeros((N+1))

    # thetau is solved backwardly
    for i in range(M-1, -1, -1):   ## backward simulation 
        # m is number of forward simulation steps
        

        # future value of thetau and thetauy
        # Notice that theta_u[i] is what we want to derive now.
        # Since we use left point to evaluate integrals and
        # theta_u[i] is currently unknown,
        # my algorithm will simply drop the integral value on the
        # first small interval, which is equivalent to use
        # zero as theta_u's initial value
        

        # parallel section
        # prange is based on OpenMP
        
        if (M-i)==1:
            theta_u[i] = 0
            theta_uy[i] = 0
        
        else:
            m = M-i
            theta_ui = theta_u[i:]
            theta_uyi = theta_uy[i:]
        
            for l in nb.prange(L):
                # standard normal values
                zmix_i = zmix[l, 1:m+1]
                z1_i = z1[l, 1:m+1]
                z2_i = z2[l, 1:m+1]
    
                # j is the jth initial state
                for j in range(0, N+1):
    
                    # initial value
                    Y_k = Y_0[j]
                    DY_k = DY_0[j]   
    
                    # To calculate $log(\xi^u)$ and $H_{\theta}^u$,
                    # it is necessary to evaluate across the full simualtion path.
                   
                    
                    Y_k1 = next_Ys(Y_k, zmix_i[0])
                    index = int((Y_k1-y_0)/delta_y)
                    index = min(index,N)
                    index = max(index,0)
                   
                    
                    #be aware of the boundary condition
                
                    theta_u_k, theta_uy_k = get_thetau(theta_ui[1], theta_uyi[1], Y_k1)
                    logxiu = next_logxiu(theta_u_k, z2_i[0])
                    theta_h_k = get_thetah(Y_k)
                    logxih = next_logxih(theta_h_k, z1_i[0])
                    xi_01 = np.exp(logxiu+logxih)
                    # finish simulating \xi^{S}
                    H_t1t2 =  sim_Ht1t2(DY_k,Y_k,theta_u_k,theta_uy_k,z1_i[0],z2_i[0])
                    #Finish simulating the product 
                    xi_temp = discretize(xi_sum,Y_k1)
                    xiH_temp = discretize(xiHtheta_sum,Y_k1)

                    xi[l,j] = xi_temp *xi_01
                    xiHtheta[l,j] = H_t1t2*xi_temp + xi_01*xiH_temp
                    
                    
            
                    
        
            # calculate thetau and thetauy for this time point
            ## add to get simulation results
            xi_sum = xi.sum(axis=0)/L
            xiHtheta_sum = xiHtheta.sum(axis=0)/L
            lst[i,0] = xi 
            lst[i,1] = xiHtheta
            lst_1[i,0] = xi_sum 
            lst_1[i,1] = xiHtheta_sum
            theta_u[i] = (gamma - 1.) * xiHtheta_sum / xi_sum
            theta_uy[i] = get_theta_uy(theta_u[i])
          

            # output thetau value to show the progress
            if i % 10 == 0 :
                print(i, N // 4, theta_u[i, N // 4])

    # the simualted value of each path at the final time step
    # is returned in order to calculate standard error
    return xi, xiHtheta


if __name__ == '__main__':

    # set a random seed
    np.random.seed(seed=0)
   
 
   
   

    lst = np.zeros((M+1,2,L,N+1))
    lst_1 = np.zeros((M+1,2,N+1))

    # initialize thetau and its first derivative with respect to y
    theta_u = np.zeros((M+1, N+1), np.float64)
    theta_uy = np.zeros((M+1, N+1), np.float64)

    # independent standard normal values
    z1 = np.random.standard_normal(size=(L, M+1))
    z2 = np.random.standard_normal(size=(L, M+1))
    print("Complete generating all the r.v.s")
    print("----------------------------")

    # start the simualtion
    xi, xiHtheta = simu_thetau(theta_u, theta_uy, z1, z2,lst,lst_1)

    # save result
    # postfix = f'CEVSV;v{v};sigmav{sigmav};M{M};N{N};L{L};deltat{delta_t};deltay{delta_y};'
    # np.savetxt(f'./Result/theta_u{postfix}.csv', theta_u, delimiter=',')
    # np.savetxt(f'./Result/xi{postfix}.csv', xi, delimiter=',')
    # np.savetxt(f'./Result/xiHtheta{postfix}.csv', xiHtheta, delimiter=',')

  

