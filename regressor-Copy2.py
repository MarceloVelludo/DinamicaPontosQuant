import DPQNova
import graphics
import numpy as np
from time import perf_counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mhz = 10**(-3)
pi = np.pi
j_1 = 280*mhz*2*pi
j_2 = 320*mhz*2*pi
bz_1 = (pi/16)*(10**3)*mhz
bz_2 = (pi/16)*(10**3)*mhz
j_12 = pi/140

t0 = perf_counter()

dpq = DPQNova.DinamicaPontosQuanticos(j_1_inicial=1, j_1_final=1, passoJ_1 = 0.5,
                 j_2_inicial=1, j_2_final=1, passoJ_2 = 0.5,
                 bz_1_inicial=0.1, bz_1_final=10, passoBz_1 = 1.0,
                 bz_2_inicial=0.1, bz_2_final=10, passoBz_2 = 1.0,
                 j_12_inicial=0.1, j_12_final=10, passoJ_12 = 1.0,
                 tInicial=1, tFinal=20, passoT=1)


dpq.make_results()


