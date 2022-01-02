import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.pyplot import figure
import plotly.express as px
import plotly.graph_objects as go
import traceback

def plotGraph(y_test,y_pred,regressorName, name, mae, mse, r2):
    y_t = y_test
    y_p = y_pred
    try:
        y_test, y_pred=zip(*random.sample(list(zip(y_test, y_pred)), 150))
    except:
        print("Gráfico com menos de 150 pontos")
        
    if max(y_test) >= max(y_pred):
        my_range = int(max(y_test))
    else:
        my_range = int(max(y_pred))
    figure(figsize=(10, 8), dpi=100)
    plt.suptitle(regressorName, y=0.95, fontsize=15)
    plt.title("%s\n"%(name)+"Média do erro absoluto: %f Média quadrada do erro: %f R2: %f" % (mae, mse, r2), fontsize=8)
    plt.scatter(range(len(y_pred)), y_pred, color='red', marker='*', alpha=0.8, label= "Predito")
    plt.scatter(range(len(y_test)), y_test, color='blue',marker='x', alpha=0.8, label= "Real")
    plt.ylabel('J_12')
    plt.xlabel('nº elemento')
    plt.legend(loc='upper right')
    plt.savefig("./data/TabelasFotos/"+'%s-%s.png'%(regressorName, name))
    plt.close()
    
    plotGraph_error(y_t,y_p,regressorName, name)
    
    return

def plotGraph_error(y_test,y_pred,regressorName, name):
    
    try:
        y_test, y_pred=zip(*random.sample(list(zip(y_test, y_pred)), 5000))
    except Exception: 
        print(Exception)
        traceback.print_exc()
        print("Gráfico com menos de 1600 pontos")
        
    if max(y_test) >= max(y_pred):
        my_range = int(max(y_test))
    else:
        my_range = int(max(y_pred))
    
    relative_error = []
    for y_p,y_r  in zip(y_pred, y_test):
        relative_error.append(np.abs((y_r-y_p)/y_r))
    
    figure(figsize=(10, 8), dpi=100)
    plt.suptitle(regressorName, y=0.95, fontsize=15)
    plt.title("%s\n"%(name), fontsize=8)
    plt.scatter(y_pred, np.array(relative_error),color='red', marker='*', alpha=0.8, label= "Predito")
#    plt.scatter(range(len(y_test)), relative_error, color='blue',marker='x', alpha=0.8, label= "Real")
    plt.ylabel('relative_error')
    plt.xlabel('real')
    plt.legend(loc='upper right')
    plt.savefig("./data/TabelasFotos/"+"Er"+'%s-%s.png'%(regressorName, name))
    plt.close()
    return
