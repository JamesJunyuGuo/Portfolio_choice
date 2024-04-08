import os
import numpy as np  # scientific computing module with Python
import numba as nb  # high performance Python compiler
import pandas as pd  # Python data analysis library

# for verification purpose; verify CEVSV 
# for KO, change model setting
# simulation parameters

M = eval(os.environ.get('M', '1000'))  # number of partition point of time t
N = eval(os.environ.get('N', '10'))  # number of partition point of state y
L = eval(os.environ.get('L', '100000'))  # number of simulation times

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

# global constant expressions
sqrt_delta_t = np.sqrt(delta_t)
sqrt_1subrho2 = np.sqrt(1 - rho ** 2)
lambda2 = lambda_ ** 2
vsub1 = v - 1

gamma_1 = (1 - gamma) / gamma
gamma_2 = (1 - gamma) / (2 * gamma)


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


@nb.vectorize('float64(float64, float64)')
def get_pitheta(thetau, y):
    r"""
    Evaluate $\pi^\theta$ given thetau and current state.

    :param thetau: value of thetau
    :param y: value of current state
    :returns: value of pitheta
    """
    return - rho / sqrt_1subrho2 / gamma / np.sqrt(y) * thetau

@nb.njit
def get_thetau_se(xi, xiHtheta):
    r"""
    Evaluate $\pi^\theta$ given thetau and current state.

    :param thetau: value of thetau
    :param y: value of current state
    :returns: value of pitheta
    """
    # number of simulation
    L = xi.shape[0]
    xi_mean = xi.sum(axis=0) / L
    xiHtheta_mean = xiHtheta.sum(axis=0) / L
    
    return (gamma - 1) * np.sqrt(
        np.sum((xiHtheta - (xiHtheta_mean / xi_mean) * xi) ** 2,
               axis=0)
    ) / (L * xi_mean)


@nb.njit
def get_pitheta_se(thetau_se, y):
    r"""
    Evaluate standard error of $\pi^\theta$
    given standard error of thetau and current state.

    :param thetau_se: standard error of thetau
    :param y: current state
    :returns: standard error of pitheta
    """
    return np.abs(rho / sqrt_1subrho2 / gamma / np.sqrt(y)) * thetau_se


@nb.njit(parallel=True)
def get_thetau_bootstrap(xi, xiHtheta, resample_time=1000):
    r"""
    Calculate Bootstrap value of thetau in order to
    report standard error of $\pi^\theta$.

    :param xi: denominator value of simualted thetau
        An array with shape (L, n). L is number of simulation
        and n is number of initial state.   
    :param xiHtheta: numerator value of simulated thetau
        An array with shape (L, n). L is number of simulation
        and n is number of initial state.
    :param resample_time: time of Bootstrap runs
    :returns thetau_bootstrap: multiple thetau derived through Bootstrap
        An array with shape (resample_time, n).
    """
    L, n = xi.shape
    
    thetau_bootstrap = np.empty((resample_time, n), np.float64)
    
    # parallel section
    for one_sample in nb.prange(resample_time):
        
        xi_tmp = np.zeros((n,), np.float64)
        xiHtheta_tmp = np.zeros((n,), np.float64)

        # for each bootstrap run, resample L times with replacement
        for l in range(L):
        
            l_rand = np.random.randint(L)
            
            for j in range(n):
        
                xi_tmp[j] += xi[l_rand, j]
                xiHtheta_tmp[j] += xiHtheta[l_rand, j]
        
        thetau_bootstrap[one_sample] = xiHtheta_tmp / xi_tmp
    
    return thetau_bootstrap


if __name__ == '__main__':

    true_thetau = get_true_theta_u.outer(np.arange(M, -1, -1) * delta_t, np.arange(0, N+1)*delta_y + y_0)
    # read file
    postfix = f'CEVSV;v{v};sigmav{sigmav};M{M};N{N};L{L};deltat{delta_t};deltay{delta_y};'
    thetau = np.loadtxt(f'theta_u{postfix}.csv', delimiter=',')
    xi = np.loadtxt(f'xi{postfix}.csv', delimiter=',') 
    xiHtheta = np.loadtxt(f'xiHtheta{postfix}.csv', delimiter=',')

    assert np.allclose(xiHtheta.sum(axis=0) / xi.sum(axis=0), thetau[0])

    # all the initial states
    ys = np.arange(0, N+1)*delta_y + y_0

    # output table
    result_df = pd.DataFrame(
        index=ys,
        columns=[
            'pitheta_simulated', 'pitheta_true_value',
            'pitheta_rlt_err', 'pitheta_se', 'pitheta_CI',
            'pitheta_se_b', 'pitheta_CI_b', 'pitheta_RMSE_b',
        ],
    )

    # simulated value of pitheta
    # derived directly from value of thetau
    # thetau[0] means we only report value with investment horizon T
    result_df['pitheta_simulated'] = get_pitheta(thetau[0], ys)

    # true value of pitheta
    # again derived directly from value of true thetau
    result_df['pitheta_true_value'] = get_pitheta(true_thetau[0], ys)

    # absolute value of relative error
    result_df['pitheta_rlt_err'] = \
        abs(result_df['pitheta_simulated'] - result_df['pitheta_true_value']) / result_df['pitheta_true_value']

    # standard error of pitheta, derived through ratio formula
    result_df['pitheta_se'] = get_pitheta_se(get_thetau_se(xi, xiHtheta), ys)  # divided by sqrt L

    # confidence interval of pitheta with asymptotic normality assumption
    result_df['pitheta_CI'] = \
    ('['
    + (result_df['pitheta_simulated'] - 1.96 * result_df['pitheta_se']).apply(lambda val: '{:.4E}'.format(val))
    + ', '
    + (result_df['pitheta_simulated'] + 1.96 * result_df['pitheta_se']).apply(lambda val: '{:.4E}'.format(val))
    + ']'
    ).values

    # Bootstrap value of thetau
    thetau_bootstrap = get_thetau_bootstrap(xi, xiHtheta, resample_time=1000)

    # standard error of simulated pitheta estimated via Bootstrap method
    # ddof=1 means degree of freedom is 1
    result_df['pitheta_se_b'] = get_pitheta_se(thetau_bootstrap.std(axis=0, ddof=1), ys)

    # confidence interval of simulated pitheta estimated via Bootstrap method
    result_df['pitheta_CI_b'] = \
    ('['
    + pd.Series(get_pitheta(np.quantile(thetau_bootstrap, 0.025, axis=0), ys)).apply(lambda val: '{:.4E}'.format(val))
    + ', '
    + pd.Series(get_pitheta(np.quantile(thetau_bootstrap, 0.975, axis=0), ys)).apply(lambda val: '{:.4E}'.format(val))
    + ']'
    ).values

    # rooted mean square error    
    result_df['pitheta_RMSE_b'] = get_pitheta_se(
        np.sqrt(np.sum((thetau_bootstrap - true_thetau[0]) ** 2, axis=0) / (len(thetau_bootstrap) - 1)),
        ys)

    # notice that:

    # 1. pitheta_true_value, pitheta_rlt_err and pitheta_RMSE_b
    # rely on true value of pitheta.
    # For GARCH-SV (v=1) and 3/2-SV (v=1.5) with intermediate
    # investment horizon (T=1, T=3),
    # the three columns are meaningless since the true
    # value estimation does not converge.

    # 2. For GARCH-SV (v=1) and 3/2-SV (v=1.5) with short
    # investment horizon (T=0.5), the true value can be
    # well estimated by the expansion approach.

    # my file "expand_thetau" only stores true value
    # at initial states from 0.1 to 0.5 with step 0.05
    # (because our final report uses these values only)
    # for initial state y = 0.11, the value of the three
    # columns are also meaningless
    result_df.to_csv(f'veri_report{postfix};.csv')
