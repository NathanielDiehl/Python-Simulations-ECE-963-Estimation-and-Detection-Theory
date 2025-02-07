
# auther: Nathan Diehl
# class: ECE 963 - Estimation and Detection Theory


from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from scipy.optimize import root
import math



def normal(mean=0, std_dev=1, y=0):
    return np.exp(-((y - mean) / std_dev) ** 2 / 2) / (std_dev * np.sqrt(2 * np.pi)) 

def laplacian(mean=0, std_dev=1, y=0):
    return np.exp(-np.sqrt(2)*np.abs((y - mean) / std_dev)) / (std_dev * np.sqrt(2))

def uniform(mean=0, std_dev=1, y=0.0):
    l = vec_in_length(y)
    l_b = np.full(l, mean - std_dev)
    u_b = np.full(l, mean + std_dev)

    return (1/(std_dev*2) * ((y > l_b) * (y < u_b)))


def vec_in_length(y):
    if type(y) != np.ndarray:
        return 1
    else:
        return len(y)


def likelihood(pdf0, pdf1, y=0):
    l = vec_in_length(y)
    return pdf1(y) / (pdf0(y)+ np.full(l, 0.0000001) )
def inverse_likelihood(pdf1, pdf2, y_i=0):
    return root(lambda x:likelihood(pdf1, pdf2, x) - y_i, 0).x.item()

def get_probability(cost, mu0, mu1, sigma, tau_prime, pdf0, pdf1):
    dy = [-1-3*sigma, tau_prime, 1+3*sigma]
    y_0 = np.linspace(dy[0], dy[1], int(1500 * np.abs(dy[1] - dy[0]) ))
    y_1 = np.linspace(dy[1], dy[2], int(1500 * np.abs(dy[2] - dy[1]) ))


    h_0 = lambda y: pdf0(y)
    h_1 = lambda y: pdf1(y)
    

    p_y = [[ np.trapz(h_0(y_0), y_0 ),
             np.trapz(h_0(y_1), y_1 )],
            [np.trapz(h_1(y_0), y_0 ),
             np.trapz(h_1(y_1), y_1 )]
            ]
    return p_y
    
def get_conditional_risk(cost, mu0, mu1, sigma, tau_prime, pdf0, pdf1):
    p_y = get_probability(cost, mu0, mu1, sigma, tau_prime, pdf0, pdf1)
    R_d = [
        cost[0][0] * p_y[0][0] + cost[1][0] * p_y[0][1],
        cost[0][1] * p_y[1][0] + cost[1][1] * p_y[1][1]
    ]
    return R_d
def get_baysian_risk(cost, mu0, mu1, sigma, tau_prime, pi, pdf0, pdf1):
    R_d =  get_conditional_risk(cost, mu0, mu1, sigma, tau_prime, pdf0, pdf1)
    r_b = pi*R_d[0] + (1-pi)*R_d[1]
    return r_b

def get_baysian_risk_plot(cost, mu0, mu1, sigma, tau_prime, pdf0, pdf1):
    pi = np.arange(0, 1, 0.01)
    R_d =  get_conditional_risk(cost, mu0, mu1, sigma, tau_prime, pdf0, pdf1)
    r_b = pi*R_d[0] + (1-pi)*R_d[1]
    return r_b


def get_min_baysian_risk_plot(cost, mu0, mu1, sigma, pdf0, pdf1):
    pi = np.arange(0, 1, 0.05)
    r_b = []
    for pi_0 in pi:
        tau        = pi_0/(1-pi_0)*(cost[1][0] - cost[0][0])/(cost[0][1] - cost[1][1])
        tau_prime  = inverse_likelihood(pdf0, pdf1, tau)
        R_d =  get_conditional_risk(cost, mu0, mu1, sigma, tau_prime, pdf0, pdf1)

        r_b.append(pi_0*R_d[0] + (1-pi_0)*R_d[1])

    return np.array(r_b)

def get_minimax_tau_prime(cost, mu0, mu1, sigma, pdf0, pdf1):
    V = get_min_baysian_risk_plot(cost, mu0, mu1, sigma, pdf0, pdf1)
    pi = np.arange(0, 1, 0.05)
    pi_0 = pi[np.argmax(V)]
    tau        = pi_0/(1-pi_0)*(cost[1][0] - cost[0][0])/(cost[0][1] - cost[1][1])
    tau_prime  = inverse_likelihood(pdf0, pdf1, tau)
    return tau_prime

def get_np_tau_prime(cost, mu0, mu1, sigma, pdf0, pdf1, alpha):
    tau_prime = 1
    d_tau_prime = 0.01
    while tau_prime > -1:
        p = get_probability(cost, mu0, mu1, sigma, tau_prime, pdf0, pdf1)
        if p[0][1] > alpha:
            return tau_prime + d_tau_prime
        tau_prime -= d_tau_prime
    return tau_prime

def get_Receiver_Operating_Characteristic(cost, mu0, mu1, sigma, pdf0, pdf1, alpha):
    tau_prime = -5
    d_tau_prime = 0.05

    P_D = []
    P_F = []
    while tau_prime < 5:
        p = get_probability(cost, mu0, mu1, sigma, tau_prime, pdf0, pdf1)
        P_F.append(p[0][1])
        P_D.append(p[1][1])
        tau_prime += d_tau_prime
    return P_F, P_D