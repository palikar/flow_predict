#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

import numpy as np


def inflow_39():
    
    inflow_avrg = np.array([[ 35.896,35.874,35.869,35.838,35.799,35.756,35.743,35.765,35.818,35.864,35.885,35.879,35.849,35.814,35.781,35.738,35.692,35.647,35.629,35.657,35.715,35.776,35.8,35.807,35.772,35.743,35.708,35.666,35.617,35.563,35.534,35.548,35.612,35.682,35.734,35.748,35.73,35.698,35.662,35.619 ],[ 35.913,35.91,35.917,35.896,35.869,35.835,35.826,35.849,35.907,35.964,35.992,35.993,35.971,35.95,35.934,35.906,35.867,35.825,35.804,35.828,35.885,35.947,35.979,35.994,35.969,35.944,35.917,35.884,35.842,35.787,35.757,35.766,35.827,35.895,35.944,35.957,35.945,35.923,35.896,35.857 ],[ 35.928,35.94,35.964,35.959,35.948,35.932,35.939,35.977,36.047,36.113,36.151,36.158,36.143,36.121,36.112,36.091,36.063,36.03,36.023,36.059,36.121,36.186,36.221,36.238,36.208,36.182,36.155,36.128,36.095,36.052,36.03,36.047,36.108,36.174,36.226,36.241,36.226,36.201,36.177,36.145 ],[ 35.943,35.962,35.998,36.006,36.002,35.988,36.004,36.058,36.141,36.217,36.269,36.295,36.292,36.279,36.27,36.255,36.232,36.203,36.204,36.254,36.337,36.419,36.47,36.507,36.501,36.492,36.472,36.444,36.414,36.372,36.356,36.385,36.465,36.552,36.619,36.648,36.65,36.64,36.623,36.585 ],[ 35.882,35.846,35.823,35.775,35.723,35.669,35.646,35.656,35.698,35.736,35.746,35.731,35.692,35.649,35.61,35.556,35.494,35.435,35.408,35.428,35.476,35.531,35.553,35.561,35.524,35.488,35.444,35.394,35.341,35.273,35.229,35.231,35.29,35.36,35.416,35.435,35.427,35.4,35.36,35.301 ],[ 35.9,35.886,35.882,35.844,35.802,35.766,35.756,35.779,35.838,35.892,35.921,35.926,35.908,35.879,35.857,35.824,35.776,35.725,35.701,35.722,35.772,35.826,35.849,35.853,35.814,35.779,35.736,35.683,35.622,35.544,35.493,35.487,35.537,35.599,35.648,35.659,35.644,35.623,35.599,35.555 ],[ 35.926,35.933,35.95,35.938,35.919,35.892,35.888,35.915,35.972,36.026,36.053,36.053,36.036,36.019,36.009,35.985,35.953,35.912,35.89,35.91,35.96,36.016,36.043,36.053,36.027,36.011,35.99,35.958,35.919,35.865,35.831,35.834,35.885,35.948,35.995,36.007,35.993,35.972,35.952,35.914 ],[ 35.93,35.939,35.956,35.941,35.917,35.882,35.87,35.891,35.947,36.0,36.025,36.02,35.992,35.96,35.932,35.892,35.846,35.795,35.762,35.775,35.822,35.876,35.896,35.901,35.863,35.831,35.795,35.748,35.698,35.638,35.605,35.613,35.67,35.735,35.782,35.79,35.772,35.743,35.708,35.663 ],]
)

    inflow_maxis = np.array([[ 51.41,51.843,52.008,52.336,52.917,53.472,53.735,53.424,53.518,54.485,55.229,57.881,58.668,59.288,60.359,61.708,63.583,64.905,65.539,66.181,66.158,65.932,66.128,65.648,65.262,65.115,64.843,67.182,70.152,71.743,72.459,72.741,72.582,72.035,71.362,71.063,70.719,70.158,68.842,68.246 ],[ 51.374,51.582,52.187,52.647,52.766,53.375,53.291,53.412,54.237,54.945,55.607,56.243,56.838,57.358,57.826,59.02,60.055,60.732,60.826,61.082,61.578,62.332,62.191,61.294,61.819,61.39,61.962,63.495,65.201,66.935,67.38,67.462,67.369,67.424,67.525,67.215,65.512,65.345,64.773,66.789 ],[ 51.757,52.133,52.404,52.523,52.735,52.73,53.111,53.538,53.627,54.136,54.296,54.748,54.799,55.172,55.675,55.653,56.197,56.366,57.087,57.626,57.841,57.535,57.936,57.923,57.687,57.715,57.693,57.467,57.303,57.745,58.21,58.099,58.207,57.929,58.85,57.846,57.992,58.533,58.405,57.77 ],[ 51.722,52.128,52.341,52.575,52.958,53.733,54.12,54.042,54.084,54.436,54.607,55.032,55.754,57.039,58.33,59.837,62.192,63.628,64.39,64.821,65.296,64.608,64.863,64.049,62.812,62.667,62.35,64.405,66.376,67.753,68.476,68.584,68.839,68.407,67.932,67.485,66.466,65.338,64.978,66.533 ],[ 52.605,53.27,53.476,53.552,53.973,53.992,54.401,55.1,56.212,56.929,56.507,56.651,56.868,57.381,59.225,60.943,62.909,64.561,65.121,65.587,66.114,65.856,65.483,65.214,64.223,62.744,64.226,66.431,68.103,69.377,70.03,70.028,69.594,68.94,68.309,68.33,68.572,66.898,65.956,65.423 ],[ 51.531,51.818,51.958,52.454,52.085,52.644,52.436,52.91,52.772,53.321,53.641,54.192,54.693,55.0,55.194,56.646,59.148,60.72,61.808,61.759,61.907,60.659,60.355,59.076,58.089,57.852,57.769,57.985,59.263,61.488,62.466,62.537,62.469,61.199,61.122,60.199,58.921,58.949,58.971,58.989 ],[ 51.622,52.022,52.104,52.117,52.497,52.479,52.85,52.834,52.841,52.875,52.533,54.201,54.336,55.154,56.34,57.574,58.232,58.637,59.046,59.095,59.031,59.555,59.246,58.359,58.143,58.282,59.938,62.026,63.234,65.108,65.998,66.537,67.69,67.649,65.839,64.302,62.801,62.36,63.229,64.838 ],[ 52.022,51.269,52.045,52.179,52.293,52.626,52.794,53.037,53.546,54.144,54.274,56.891,57.047,57.38,59.075,59.807,61.086,61.97,62.126,62.316,62.845,62.189,62.383,61.767,62.158,62.791,64.168,65.869,66.992,67.947,68.465,68.559,68.677,68.639,68.475,67.953,67.179,67.127,68.131,68.95 ],]
)



    c_avrg_avrgs = np.mean(inflow_avrg, axis=0)
    c_avrg_stds = np.std(inflow_avrg, axis=0)

    c_max_avrgs = np.mean(inflow_maxis, axis=0)
    c_max_stds = np.std(inflow_maxis, axis=0)

    
    for i in range(0, 40, 5):
        print('{:.2f}({:.2f}) & {:.2f}({:.2f})'.format(c_avrg_avrgs[i], c_avrg_stds[i], c_max_avrgs[i], c_max_stds[i]))


def inflow_8():
    
    inflow_avrg = np.array([[ 34.262,34.218,34.128,34.096,34.015,33.82,33.743,33.934,34.103,34.014,33.919,34.038,34.126,33.969,33.723,33.73,33.994,34.242,34.301,34.217,34.167,34.145,33.992,33.721,33.651,33.848,34.111,34.215,34.139,34.047,33.986,33.829,33.616,33.657,33.978,34.309,34.396,34.256,34.105,33.971 ],[ 34.287,34.274,34.212,34.203,34.148,33.965,33.893,34.095,34.28,34.209,34.132,34.28,34.393,34.259,34.016,34.009,34.265,34.521,34.586,34.512,34.484,34.483,34.352,34.095,34.003,34.175,34.429,34.541,34.467,34.388,34.354,34.221,34.027,34.068,34.368,34.69,34.765,34.621,34.466,34.353 ],[ 34.28,34.255,34.183,34.167,34.11,33.946,33.872,34.069,34.261,34.208,34.142,34.293,34.399,34.272,34.024,33.993,34.225,34.481,34.558,34.522,34.516,34.522,34.394,34.112,33.988,34.119,34.344,34.441,34.431,34.429,34.436,34.311,34.07,34.051,34.306,34.551,34.593,34.497,34.453,34.425 ],[ 34.303,34.294,34.224,34.213,34.151,33.971,33.896,34.098,34.296,34.236,34.161,34.287,34.383,34.238,33.983,33.962,34.208,34.473,34.562,34.509,34.463,34.427,34.269,33.99,33.898,34.064,34.308,34.454,34.445,34.402,34.334,34.13,33.897,33.919,34.208,34.495,34.578,34.524,34.473,34.369 ],[ 34.224,34.158,34.043,33.975,33.866,33.641,33.525,33.686,33.852,33.768,33.677,33.789,33.852,33.681,33.416,33.397,33.639,33.897,34.005,33.959,33.903,33.854,33.665,33.386,33.311,33.479,33.724,33.868,33.88,33.84,33.763,33.544,33.301,33.349,33.654,33.951,34.037,33.971,33.93,33.828 ],[ 34.251,34.184,34.091,34.047,33.957,33.752,33.648,33.809,33.971,33.888,33.795,33.928,34.028,33.869,33.611,33.591,33.828,34.068,34.123,34.048,34.008,34.006,33.846,33.563,33.481,33.652,33.886,33.982,33.914,33.848,33.818,33.665,33.441,33.486,33.803,34.1,34.157,34.019,33.902,33.797 ],[ 34.27,34.239,34.151,34.116,34.043,33.856,33.77,33.955,34.135,34.061,33.993,34.128,34.226,34.084,33.828,33.808,34.055,34.3,34.35,34.281,34.259,34.256,34.105,33.837,33.746,33.922,34.17,34.257,34.183,34.144,34.136,33.994,33.778,33.809,34.117,34.406,34.429,34.254,34.155,34.114 ],[ 34.282,34.256,34.187,34.187,34.134,33.949,33.864,34.045,34.237,34.178,34.101,34.238,34.346,34.207,33.948,33.907,34.139,34.394,34.481,34.421,34.389,34.377,34.247,33.996,33.879,34.009,34.248,34.38,34.352,34.293,34.245,34.088,33.904,33.94,34.209,34.498,34.581,34.486,34.384,34.269 ],]
)

    inflow_maxis = np.array([[ 52.408,52.301,55.468,59.962,59.233,62.816,63.376,65.326,66.148,66.902,67.422,66.664,67.511,70.483,76.476,78.998,79.194,79.245,79.096,78.746,78.033,77.394,78.09,85.18,85.394,83.739,82.375,81.165,80.211,80.064,79.554,84.507,87.467,88.421,88.489,83.515,81.899,83.218,84.252,79.862 ],[ 52.221,53.056,53.952,58.156,57.825,59.974,61.942,63.694,65.449,66.613,67.761,66.863,69.582,71.716,77.429,79.763,79.25,79.353,79.358,79.007,78.804,78.124,82.868,86.63,85.568,83.257,81.877,81.017,81.083,80.237,82.039,87.241,88.378,88.429,88.219,83.836,81.94,85.125,87.001,84.732 ],[ 52.954,53.617,53.894,55.878,55.313,58.198,59.076,60.876,61.164,62.319,62.813,62.011,61.321,63.153,67.0,69.58,69.849,69.989,70.659,70.808,69.969,68.245,71.335,77.866,79.624,79.288,78.995,78.49,77.658,76.678,76.104,79.023,86.237,85.797,83.448,81.868,80.782,79.907,79.88,79.488 ],[ 52.717,52.664,53.856,58.077,58.661,61.818,63.758,65.069,65.927,67.364,67.71,66.986,65.819,70.073,76.371,78.102,77.932,77.881,77.519,77.059,76.182,75.437,77.025,84.382,84.158,85.198,80.498,79.271,79.816,81.416,76.737,82.567,85.467,88.633,88.748,84.233,80.44,83.554,86.738,81.74 ],[ 53.255,53.879,55.16,59.351,58.619,60.921,62.684,64.703,65.03,66.642,66.584,65.474,66.234,69.789,77.286,80.788,79.676,78.928,77.706,76.893,77.16,76.132,79.541,84.28,86.266,88.301,85.181,78.99,83.151,86.893,81.977,80.09,83.331,87.954,88.335,87.943,80.52,83.761,87.606,87.544 ],[ 52.104,52.313,55.005,58.209,58.43,61.564,63.166,64.767,65.88,67.669,68.386,68.502,68.258,68.739,72.681,75.062,75.927,76.006,77.124,77.247,76.156,75.933,75.171,79.022,81.75,80.806,79.887,80.214,79.314,79.033,78.437,78.314,84.257,83.644,82.252,80.891,79.744,79.451,79.039,78.498 ],[ 52.919,53.451,54.03,57.754,57.831,57.85,59.474,60.036,60.802,62.033,62.469,61.457,64.985,66.259,69.426,72.021,72.012,71.767,71.729,71.523,70.639,72.372,75.623,80.749,81.993,80.04,79.961,79.23,77.726,76.335,77.353,82.006,86.743,85.465,83.316,81.463,79.911,80.713,78.338,80.965 ],[ 53.168,53.42,55.919,59.378,58.631,60.157,61.368,63.17,63.198,64.31,65.148,66.075,68.309,71.476,75.808,76.915,76.844,76.753,77.171,77.218,76.163,76.646,80.6,84.429,85.324,86.933,81.384,79.913,81.828,82.442,81.026,85.128,87.495,88.488,88.432,84.409,80.51,85.425,87.408,83.523 ],]
)



    c_avrg_avrgs = np.mean(inflow_avrg, axis=0)
    c_avrg_stds = np.std(inflow_avrg, axis=0)

    c_max_avrgs = np.mean(inflow_maxis, axis=0)
    c_max_stds = np.std(inflow_maxis, axis=0)

    
    for i in range(0, 40, 5):
        print('{:.2f}({:.2f}) & {:.2f}({:.2f})'.format(c_avrg_avrgs[i], c_avrg_stds[i], c_max_avrgs[i], c_max_stds[i]))        


def fluid_25():
    
    inflow_avrg = np.array([[ 35.831,35.886,36.051,36.131,36.154,36.148,36.114,36.044,35.925,35.751,35.505,35.341,35.264,35.292,35.383,35.488,35.537,35.59,35.552,35.456,35.387,35.401,35.462,35.495,35.464,35.314,35.119,34.973,34.961,35.09,35.318,35.538,35.628,35.682,35.611,35.515,35.497,35.584,35.698,35.733 ],[ 35.855,35.953,36.136,36.23,36.277,36.29,36.276,36.212,36.086,35.92,35.676,35.522,35.446,35.473,35.549,35.645,35.679,35.724,35.675,35.567,35.489,35.504,35.575,35.622,35.59,35.428,35.228,35.074,35.066,35.192,35.406,35.6,35.652,35.684,35.603,35.501,35.468,35.537,35.641,35.679 ],[ 35.91,36.048,36.267,36.395,36.479,36.538,36.564,36.538,36.444,36.29,36.082,35.948,35.899,35.955,36.055,36.18,36.234,36.301,36.269,36.177,36.113,36.151,36.247,36.307,36.285,36.14,35.95,35.813,35.82,35.965,36.201,36.41,36.476,36.509,36.43,36.327,36.3,36.379,36.492,36.537 ],]
)

    inflow_maxis = np.array([[ 51.541,51.828,52.387,53.326,53.804,53.709,53.63,53.91,54.07,54.188,54.172,53.585,53.249,53.181,54.639,55.94,56.144,56.859,57.109,56.354,56.448,56.522,56.708,56.904,57.271,57.489,57.593,57.241,57.035,56.538,56.528,57.406,60.021,61.196,61.71,61.177,60.706,61.785,62.987,64.04 ],[ 52.109,52.497,53.238,54.499,54.2,54.016,54.735,55.434,56.017,55.631,55.474,56.063,56.91,57.062,57.729,58.385,57.934,59.04,60.066,60.878,60.626,60.827,61.178,61.609,62.364,61.482,63.132,63.296,63.341,63.078,62.517,62.545,62.281,62.536,62.731,63.137,63.925,64.428,64.375,64.718 ],[ 52.457,52.79,54.529,55.233,56.0,55.73,56.705,57.102,57.372,57.869,57.95,57.826,58.988,59.561,60.216,61.08,61.168,61.807,62.341,62.975,62.533,62.994,63.311,64.015,64.089,63.62,63.944,64.182,63.902,64.286,64.444,64.911,64.987,65.053,64.933,65.021,64.941,65.189,65.177,65.516 ],]
)



    c_avrg_avrgs = np.mean(inflow_avrg, axis=0)
    c_avrg_stds = np.std(inflow_avrg, axis=0)

    c_max_avrgs = np.mean(inflow_maxis, axis=0)
    c_max_stds = np.std(inflow_maxis, axis=0)

    
    for i in range(0, 40, 5):
        print('{:.2f}({:.2f}) & {:.2f}({:.2f})'.format(c_avrg_avrgs[i], c_avrg_stds[i], c_max_avrgs[i], c_max_stds[i]))


def fluid_200():
    
    inflow_avrg = np.array([[ 35.27,35.275,35.28,35.285,35.291,35.297,35.303,35.31,35.317,35.324,35.332,35.34,35.348,35.357,35.366,35.375,35.384,35.394,35.404,35.413,35.423,35.434,35.444,35.454,35.464,35.474,35.484,35.494,35.504,35.514,35.523,35.533,35.542,35.551,35.56,35.569,35.578,35.586,35.595,35.603 ],[ 35.265,35.264,35.262,35.261,35.259,35.257,35.255,35.253,35.251,35.25,35.248,35.246,35.244,35.243,35.241,35.24,35.239,35.238,35.237,35.236,35.235,35.235,35.235,35.235,35.235,35.236,35.237,35.237,35.239,35.24,35.241,35.243,35.244,35.246,35.248,35.25,35.251,35.253,35.256,35.258 ],[ 35.265,35.262,35.26,35.257,35.255,35.254,35.252,35.25,35.249,35.248,35.247,35.246,35.245,35.245,35.244,35.244,35.243,35.243,35.243,35.243,35.243,35.243,35.243,35.243,35.243,35.243,35.244,35.244,35.244,35.245,35.245,35.246,35.246,35.247,35.247,35.248,35.248,35.249,35.249,35.25 ],]
)

    inflow_maxis = np.array([[ 50.808,51.157,50.73,50.798,50.888,50.939,50.971,50.995,51.016,51.036,51.058,51.087,51.256,51.606,52.066,52.337,52.558,52.754,53.241,53.849,54.139,54.335,54.494,54.64,54.789,54.937,55.083,55.204,55.295,55.363,55.417,55.462,55.5,55.528,55.566,55.664,55.755,55.833,55.901,55.957 ],[ 54.17,56.026,57.4,58.081,58.467,58.708,58.867,58.974,59.047,59.097,59.131,59.153,59.165,59.17,59.169,59.166,59.161,59.157,59.153,59.149,59.146,59.435,59.812,60.174,60.528,60.874,61.216,61.552,61.882,62.206,62.522,62.83,63.112,63.366,63.594,63.799,63.981,64.136,64.24,64.317 ],[ 51.503,52.0,52.322,52.594,52.846,53.051,53.353,53.726,54.026,54.275,54.493,54.698,54.902,55.108,55.313,55.514,55.713,55.911,56.105,56.297,56.482,56.632,56.712,56.76,56.787,56.804,56.816,56.824,56.832,56.838,56.974,57.232,57.464,57.682,57.876,58.041,58.182,58.302,58.402,58.485 ],]
)



    c_avrg_avrgs = np.mean(inflow_avrg, axis=0)
    c_avrg_stds = np.std(inflow_avrg, axis=0)

    c_max_avrgs = np.mean(inflow_maxis, axis=0)
    c_max_stds = np.std(inflow_maxis, axis=0)

    
    for i in range(0, 40, 5):
        print('{:.2f}({:.2f}) & {:.2f}({:.2f})'.format(c_avrg_avrgs[i], c_avrg_stds[i], c_max_avrgs[i], c_max_stds[i]))        


def main():

    # print('Inflow 8')
    # inflow_8()
    # print('Inflow 39')
    # inflow_39()

    print('Fluid 25')
    fluid_25()
    print('Fluid 200')
    fluid_200()
    
    

if __name__ == '__main__':
    main()
