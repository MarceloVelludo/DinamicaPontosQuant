import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.pyplot import figure



def plotGraph(y_test,y_pred,regressorName, name, mae, mse, r2):
    try:
        y_test, y_pred=zip(*random.sample(list(zip(y_test, y_pred)), 400))
    except:
        print("Gráfico com menos de 400 pontos")
        
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
    return

