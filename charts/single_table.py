#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

import numpy as np






def main():

    
    p_const_avrg = np.array([1.0414251968503936 , 0.6457401574803151 , 0.42076377952755895 , 0.2865590551181102 ,])
    p_const_max = np.array([16.123661417322836 , 16.308669291338582 , 11.390377952755905 , 14.41744881889764 ])
    np_const_avrg = np.array([0.32077165354330706 , 0.5521338582677168 , 0.3627007874015747 , 1.6469842519685036])
    np_const_max = np.array([30.49359055118111 , 29.123480314960634 , 75.47813385826771 , 41.04614173228347])
    
    p_speed_avrg = np.array([0.5161387225548902 , 0.5086556886227545 , 0.534815369261477 , 0.6252145708582835])
    p_speed_max = np.array([31.709387225548905 , 32.00352095808383 , 31.838751497005987 , 33.08689720558882])
    np_speed_avrg = np.array([0.8764880239520958 , 0.8413493013972057 , 1.0208942115768462 , 0.97850499001996 ])
    np_speed_max = np.array([46.87356187624751 , 43.75898702594811 , 46.5552864271457 , 45.54569660678643])

    p_fluid_avrg = np.array([0.41133193702152243 , 27.280076848998455])
    p_fluid_max = np.array([14.537741441571573 , 99.45428139445299])
    np_fluid_avrg = np.array([0.42235304314329736 , 0.5787415254237288 , 0.33393451463790447 , 0.30568432203389834 , 0.3313731041456016 , 0.3340954788386537 , 0.31267950693374424 ])
    np_fluid_max = np.array([16.124258474576273 , 14.990394260400619 , 13.283522534668721 , 19.36700442989214 , 11.396986710963457 , 11.501868842987145 , 16.20593278120185 ])

    print("{:.3}({:.3})&{:.3}({:.3})&{:.3}({:.3})&{:.3}({:.3})".format(
        np.mean(p_const_max),
        np.var(p_const_max),
        np.mean(np_const_max),
        np.var(np_const_max),
        np.mean(p_const_avrg),
        np.var(p_const_avrg),
        np.mean(np_const_avrg),
        np.var(np_const_avrg)))

    
    print("{:.3}({:.3})&{:.3}({:.3})&{:.3}({:.3})&{:.3}({:.3})".format(
        np.mean(p_speed_max),
        np.var(p_speed_max),
        np.mean(np_speed_max),
        np.var(np_speed_max),
        np.mean(p_speed_avrg),
        np.var(p_speed_avrg),
        np.mean(np_speed_avrg),
        np.var(np_speed_avrg)))

    print("{:.3}({:.3})&{:.3}({:.3})&{:.3}({:.3})&{:.3}({:.3})".format(
        np.mean(p_fluid_max),
        np.var(p_fluid_max),
        np.mean(np_fluid_max),
        np.var(np_fluid_max),
        np.mean(p_fluid_avrg),
        np.var(p_fluid_avrg),
        np.mean(np_fluid_avrg),
        np.var(np_fluid_avrg)))
    

if __name__ == '__main__':
    main()
