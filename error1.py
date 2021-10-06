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

dpq = DPQNova.DinamicaPontosQuanticos(j_1_inicial= j_1*0.8, j_1_final=j_1*1.2, passoJ_1 = 0.1,
                                      j_2_inicial=j_2*0.8, j_2_final=j_2*1.2, passoJ_2 = 0.1,
                 bz_1_inicial=bz_1*0.8, bz_1_final=bz_1*1*2, passoBz_1 = 0.01,
                 bz_2_inicial=bz_1*0.8, bz_2_final=bz_2*1.2, passoBz_2 = 0.01,
                 j_12_inicial=j_12*0.8, j_12_final=j_12*1.2, passoJ_12 = 0.0001,
                 tInicial=1, tFinal=20, passoT=1)


df = dpq.criaDataFrame()
t1 = perf_counter()

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,5:], df.iloc[:,4], test_size=0.3, random_state= 0)
reg = ExtraTreesRegressor(n_estimators=100, random_state=0, n_jobs= -1).fit(X_train, y_train)

y_train_pred = reg.predict(X_train)
t2 = perf_counter()
relative_error_1 = 0
relative_error_2 = 0

mae = mean_absolute_error(y_train, y_train_pred)
mse = mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_train,y_train_pred)

for y_p, y_t in zip(y_train_pred, y_train):
    relative_error_1 += np.abs(1-y_p)/y_t
    relative_error_2 += np.abs((y_t-y_p)/y_t)
    

with open("Speed.txt", "a+") as text_file:
    text_file.write("===="*5)
    text_file.write("\n%s"%(dpq.name_comp))
    text_file.write("Tempo tabela: %f \nTempo regressor: %f" % (t1-t0,t2-t1))


with open("ResultadosTreino.txt", "a+") as text_file:
    text_file.write("===="*5)
    text_file.write("\n%s"%(dpq.name_comp))
    text_file.write("\nMédia do erro absoluto: %f \nMédia quadrada do erro: %f \nR2: %f\nrelative_error_1: %f \nrelative_error_2: %f" % (mae, mse, r2, relative_error_1, relative_error_2))
    
relative_error_1 = 0
relative_error_2 = 0
graphics.plotGraph(y_train,y_train_pred, "Extra Trees Regressor Train",dpq.name, mae, mse, r2)
    
y_test_pred = reg.predict(X_test)

for y_p, y_t in zip(y_test_pred, y_test):
    relative_error_1 += np.abs(1-y_p)/y_t
    relative_error_2 += np.abs((y_t-y_p)/y_t)
    
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test,y_test_pred)

with open("ResultadosTest.txt", "a+") as text_file:
    text_file.write("===="*5)
    text_file.write("\n%s"%(dpq.name_comp))
    text_file.write("\nMédia do erro absoluto: %f \nMédia quadrada do erro: %f \nR2: %f\nrelative_error_1: %f \nrelative_error_2: %f" % (mae, mse, r2, relative_error_1, relative_error_2))

graphics.plotGraph(y_test,y_test_pred, "Extra Trees Regressor Test",dpq.name_comp ,mae, mse, r2)

dpq.saveDataFrame()

dpq.save_Y(y_test, y_test_pred, len(y_test))

