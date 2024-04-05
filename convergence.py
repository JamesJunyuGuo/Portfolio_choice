#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:38:35 2023

@author: jamesguo
"""

import os
import numpy as np  # scientific computing module with Python
import numba as nb  # high performance Python compiler
import pandas as pd  # Python data analysis library
# for verification purpose; verify CEVSV 
# for KO, change model setting
# simulation parameters

M = eval(os.environ.get('M', '100'))  # number of partition point of time t
N = eval(os.environ.get('N', '10'))  # number of partition point of state y
L = eval(os.environ.get('L', '1000'))  # number of simulation times


t_0 = 0  # start time point
t_M = eval(os.environ.get('T_M', '1'))  # end time point T
y_0 = eval(os.environ.get('Y_0', '0.1'))  # minimal value of initial state y
y_N = eval(os.environ.get('Y_N', '0.2'))  # maximal value of initial state y

delta_t = (t_M - t_0)/ M
delta_y = (y_N - y_0) / N
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

# global constant expressions
sqrt_delta_t = np.sqrt(delta_t)
sqrt_1subrho2 = np.sqrt(1 - rho ** 2)
lambda2 = lambda_ ** 2
vsub1 = v - 1

gamma_1 = (1 - gamma) / gamma
gamma_2 = (1 - gamma) / (2 * gamma)

print(
    f'Parameter Sets: \
v = {v}; r = {r}; kappa = {kappa}; \
theta = {theta}; sigmav = {sigmav}; \
lambda_ = {lambda_}; rho = {rho}; gamma = {gamma}'
)



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


@nb.njit(   ## main part 
    parallel=True,  # parallel iteration based on OpenMP
    fastmath=True,  # enable LLVM fastmath to speed up
    nogil=True  # release Python global interpreter lock (GIL) to speed up
)
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

        # parallel section
        # prange is based on OpenMP
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
                logxiu = 0.
                Hthetau = 0.
                for k in range(m-1):
                    # thetau and thetauy at simualted future state value
                    theta_u_k, theta_uy_k = get_thetau(theta_ui[k], theta_uyi[k], Y_k)

                    logxiu += next_logxiu(theta_u_k, z2_i[k])
                    Hthetau += next_Hthetau(theta_u_k, theta_uy_k, DY_k, z2_i[k])
                    
                    # move state variable Y and Mallivin derivative DY to the next time point
                    DY_k = next_DYs(DY_k, Y_k, zmix_i[k])
                    Y_k = next_Ys(Y_k, zmix_i[k])

                # theta_h at the final step
                # used to increment $log(\xi^h)$ and $H_{\theta}^h$
                theta_h_k = get_thetah(Y_k)
                logxih[l, j] = logxih[l, j] + next_logxih(theta_h_k, z1_i[-1])
                Hthetah[l, j] = Hthetah[l, j] + next_Hthetah(theta_h_k, DY_k, z1_i[-1])
                
                xi_tmp = np.exp(logxih[l, j] + logxiu)
                Htheta_tmp = Hthetah[l, j] + Hthetau

                # store simualted numerator value and denominator value of thetau
                xi[l, j] =  xi_tmp
                xiHtheta[l, j] = xi_tmp * Htheta_tmp
        
        # calculate thetau and thetauy for this time point
        ## add to get simulation results
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



if v == 0.5:
    @nb.vectorize('float64(float64, float64)')
    def get_true_theta_u(tau, y):
        """
        Evaluate theta_u by the closed-form solution.
        Use true theta_u as the benchmark for our reports.

        :param tau: T - t, time to maturity
        :param y: initial state
        """
        deltav = - (1 - gamma) * lambda_**2 / (2 * gamma**2)
        ktilde = kappa - (1 - gamma) * lambda_ * sigmav * rho / gamma
        xiv = np.sqrt(ktilde**2 + 2 * deltav * (rho**2 + gamma*(1 - rho**2)) * sigmav**2)

        d_tau = -2 *deltav * (np.exp(xiv*tau) - 1) / ((ktilde + xiv)*(np.exp(xiv*tau) - 1) + 2 * xiv)

        return - sqrt_1subrho2 * gamma * sigmav * np.sqrt(y) * d_tau
else:
    # Only Heston's model (v=0.5) has closed-form solution.
    # For GARCH-SV model and 3/2-SV model, take the approximation
    # value given by expansion algorithm of Chenxu Li (2019)
    # as the true value.
    # For small horizon (T<=0.5), this will work since expansion
    # algorithm achieves far higher order accuracy than the simulation.
    # For intermediate time horizon (T = 1),
    # the expansion algorithm does not converge well and the
    # accuracy report is meaningless.
    f_name = f'expan_theta_uCEVSV;v{v};sigmav{sigmav}.csv'
    expand_theta_u = np.array(
        pd.read_csv(
            f_name, header=None, dtype='object'
        ).applymap(
            # make the floating number output by mathematica
            # readable via Python
            lambda x: list(eval(
                x[1:-1].replace('*^', 'e')))
        ).values.tolist()
    )

    @nb.vectorize('float64(float64, float64)')
    def get_true_theta_u(tau, y):
        """
        Evaluate thetau by expansion approximation of Chenxu Li (2019).
        Use true thetau as the benchmark for our reports.

        :param tau: T - t, time to maturity
        :param y: initial state
        returns: value of true thetau
        """
        if tau == 0:
            return 0
        
        # this is hard coded
        # expand_theta_u is a 3-dimensional array
        # the last index of expand_theta_u represents different orders of expansion
        # and I stored value of the first 20 orders
        # The 20th order acheives 10^(-10) accuracy already under Heston's model given T=0.5
        # expand_theta_u[:, :, -1] represents value of the last order (order=20)
        # round(tau*1000-1) is the index of time since the time step is 0.001
        # round(round(y - 0.1, 5) / 0.05) is the index of state since y ranges from
        # 0.1 to 0.5 with step 0.05
        return expand_theta_u[:, :, -1][round(tau*1000-1), round(round(y - 0.1, 5) / 0.05)]

def convergence_test(M=100, L=1000, N =10):
    delta_t = (t_M - t_0)/ M
    delta_y = (y_N - y_0) / N
    assert np.allclose(t_M, t_0 + delta_t * M)
    assert np.allclose(y_N, y_0 + delta_y * N)
    
    
    
    return 

if __name__ == '__main__':

    true_thetau = get_true_theta_u.outer(np.arange(M, -1, -1) * delta_t, np.arange(0, N+1)*delta_y + y_0)
    # read file
    # true_thetau = get_true_theta_u.outer(np.arange(M, -1, -1) * delta_t, np.arange(0, N+1)*delta_y + y_0)
    # postfix = f'CEVSV;v{v};sigmav{sigmav};M{M};N{N};L{L};deltat{delta_t};deltay{delta_y};'
    # theta_u = np.zeros((M+1, N+1), np.float64)
    # theta_uy = np.zeros((M+1, N+1), np.float64)

    # # independent standard normal values
    M_list = [500,500, 1000, 1000, 5000]
    M_list = np.array(M_list)
    L_list = M_list*2
    L_list[[1,3]] *= 5
    # M_list = [1000,1000]
    # M_list = np.array(M_list)
    # L_list =[100000, 500000]
    
    for m,l in zip(M_list,L_list):
        M = int(m)
        L = int(l)
        
  
        delta_t = (t_M - t_0)/ M
        sqrt_delta_t = np.sqrt(delta_t)
        delta_y = (y_N - y_0) / N
        assert np.allclose(t_M, t_0 + delta_t * M)
        assert np.allclose(y_N, y_0 + delta_y * N)
        print(M,L,delta_t)
        true_thetau = get_true_theta_u.outer(np.arange(M, -1, -1) * delta_t, np.arange(0, N+1)*delta_y + y_0)
        theta_u = np.zeros((M+1, N+1), np.float64)
        theta_uy = np.zeros((M+1, N+1), np.float64)
        z1 = np.random.standard_normal(size=(L, M+1))
        z2 = np.random.standard_normal(size=(L, M+1))
        print("Complete generating all the r.v.s")
        print("----------------------------")
        xi, xiHtheta = simu_thetau(theta_u, theta_uy, z1, z2)
        delta_thetau = theta_u - true_thetau
        np.save(f'./Convergence_Result/Delta_thetau;M{M};L{L}.npy',delta_thetau)
        np.save(f'./Convergence_Result/xi;M{M};L{L}.npy',xi)
        np.save(f'./Convergence_Result/xiHtheta;M{M};L{L}.npy',xiHtheta)

        
        
        
        

        
        

