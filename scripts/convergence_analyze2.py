#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:43:16 2023

@author: jamesguo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:22:44 2023

@author: jamesguo
"""

import numpy as np 
import pandas as pd
M_list = [500,500, 1000, 1000, 5000,5000]
M_list = np.array(M_list)
L_list = M_list*2
L_list[[1,3,5]] *= 5
N = 10
result = []
i = 0
result = []
for M,L in zip(M_list,L_list):
    delta_thetau = np.load(f'./Convergence_Result/Delta_thetau;M{M};L{L}.npy')
    result.append( np.max(np.abs(delta_thetau),axis=1))
    
result = pd.DataFrame(result)
result.to_csv('./Convergence_Result/convergence_plot.csv',index=None)




    