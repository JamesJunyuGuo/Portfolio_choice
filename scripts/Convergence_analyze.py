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
# i = 0
result = pd.DataFrame(columns=['M','N','L','RMSE1','RMSE2','Bias1','Bias2'])
for M,L in zip(M_list,L_list):
    delta_thetau = np.load(f'./Convergence_Result/Delta_thetau;M{M};L{L}.npy')
    xi = np.load(f'./Convergence_Result/xi;M{M};L{L}.npy')
    xiHtheta = np.load(f'./Convergence_Result/xiHtheta;M{M};L{L}.npy')
   
#     result.append( np.max(np.abs(delta_thetau),axis=1))
    
# result = pd.DataFrame(result)
# result.to_csv('./Convergence_Result/convergence_plot.csv',index=None)
    rmse1 = np.max(np.std(xiHtheta,axis=0))
    rmse2 = np.max(np.std(xi,axis=0))
    temp = np.zeros((7))
    delta_1 = np.mean(np.abs(delta_thetau))
    delta_2 = np.sqrt(L)*delta_1
    temp = [M,N,L,rmse1,rmse2,delta_1,delta_2]
    result.loc[len(result)] = temp
    


result.to_csv("./Convergence_Result/Table_Convergence.csv",index=None)
    