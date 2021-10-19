#Para construir a tabela(dataframe) que será usada para treinar o modelo vamos usar o Pandas.
import numpy as np                                     
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools
import random
import graphics

from itertools import product
from numba import jit, vectorize 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sympy.physics.quantum import TensorProduct
from time import perf_counter 
from scipy.linalg import expm
from cmath import  *
from decimal import *
from time import perf_counter
from multiprocessing import Pool
#from math import *

#A classe DinamicaPontosQuanticos calcula a dinamica no intervalo desejado e cria a tabela com os resultados.
class DinamicaPontosQuanticos:
    
    def countDecimal2(self):
        passos = [self.passoJ_1, self.passoJ_2, self.passoBz_1,self.passoBz_2, self.passoJ_12, self.passoT]
        decimals = np.array([])
        for passo in passos:
            decimals = np.append(decimals, 10**(-Decimal(str(passo)).as_tuple().exponent))
        return (decimals[0], decimals[1], decimals[2], decimals[3], decimals[4], decimals[5]) 
  
    def __init__(self, j_1_inicial=0, j_1_final=1, passoJ_1 = 0.5,
                 j_2_inicial=0, j_2_final=1, passoJ_2 = 0.5,
                 bz_1_inicial=0, bz_1_final=1, passoBz_1 = 0.5,
                 bz_2_inicial=0, bz_2_final=1, passoBz_2 = 0.5,
                 j_12_inicial=0, j_12_final=10, passoJ_12 = 0.1,
                 tInicial=1, tFinal=20, passoT=1):
        self.j_1_inicial = j_1_inicial
        self.j_1_final = j_1_final
        self.j_2_inicial = j_2_inicial
        self.j_2_final = j_2_final
        self.bz_1_inicial = bz_1_inicial
        self.bz_1_final = bz_1_final
        self.bz_2_inicial = bz_2_inicial
        self.bz_2_final = bz_2_final
        self.j_12_inicial = j_12_inicial
        self.j_12_final = j_12_final
        self.tInicial = tInicial
        self.tFinal = tFinal
        self.passoJ_1 = passoJ_1 
        self.passoJ_2 = passoJ_2 
        self.passoBz_1 = passoBz_1
        self.passoBz_2 = passoBz_2
        self.passoJ_12 = passoJ_12
        self.passoT = passoT
        
        #Para criar relatorios
        self.name = str("["+str(self.j_1_inicial)[:4]+":"+str(self.j_1_final)[:4]+":"+str(self.passoJ_1)[:4]+"]"+"["+str(self.j_2_inicial)[:4]+":"+str(self.j_2_final)[:4]+":"+str(self.passoJ_2)[:4]+"]"+"["+str(self.bz_1_inicial)[:4]+":"+str(self.bz_1_final)[:4]+":"+str(self.passoBz_1)[:4]+"]"+"["+str(self.bz_2_inicial)[:4]+":"+str(self.bz_2_final)[:4]+":"+str(self.passoBz_2)[:4]+"]"+"["+str(self.j_12_inicial)[:4]+":"+str(self.j_12_final)[:4]+":"+str(self.passoJ_12)[:4]+"]"+"["+str(self.tInicial)[:3]+":"+str(self.tFinal)[:3]+":"+str(self.passoT)[:4]+"]")
        
        self.name_comp = str("["+str(self.j_1_inicial)+":"+str(self.j_1_final)+":"+str(self.passoJ_1)+"]"+"["+str(self.j_2_inicial)+":"+str(self.j_2_final)+":"+str(self.passoJ_2)+"]"+"["+str(self.bz_1_inicial)+":"+str(self.bz_1_final)+":"+str(self.passoBz_1)+"]"+"["+str(self.bz_2_inicial)+":"+str(self.bz_2_final)+":"+str(self.passoBz_2)+"]"+"["+str(self.j_12_inicial)+":"+str(self.j_12_final)+":"+str(self.passoJ_12)+"]"+"["+str(self.tInicial)+":"+str(self.tFinal)+":"+str(self.passoT)+"]")
        
        #roInicial
        #UpUp
        #self.ro0 = np.array([[1,0,0,0], [0,0,0,0], [0,0,0,0],[0,0,0,0]])
        self.ro0 = np.array([[0.25,0.25,0.25,0.25], [0.25,0.25,0.25,0.25], [0.25,0.25,0.25,0.25],[0.25,0.25,0.25,0.25]])
        #Criando as matrizes de Pauli-X, Pauli-Y, e Pauli-Z.
        self.sigmaX = np.array([[0, 1], [1, 0]])
        self.sigmaY = np.array([[0, -1j], [1j, 0]])
        self.sigmaZ = np.array([[1, 0], [0, -1]])
        
        #Matriz identidade.
        self.ident = np.identity(2)
        
        #Algumas constantes que são usadas diversas vezes para os calculos.
        self.tensorProductIdentSigX = TensorProduct(self.ident, self.sigmaX)
        self.tensorProductSigXIdent = TensorProduct(self.sigmaX, self.ident)
        self.tensorProductIdentSigY = TensorProduct(self.ident, self.sigmaY)
        self.tensorProductSigYIdent = TensorProduct(self.sigmaY, self.ident)
        self.tensorProductIdentSigZ = TensorProduct(self.ident, self.sigmaZ)
        self.tensorProductSigZIdent = TensorProduct(self.sigmaZ, self.ident)
        self.tensorProductSigZIdentSoma = TensorProduct(self.sigmaZ, self.ident) + TensorProduct(self.ident, self.sigmaZ)
        self.tensorProductSigZSigZ = TensorProduct(self.sigmaZ, self.sigmaZ)
        
        decimalJ_1, decimalJ_2, decimalBz_1, decimalBz_2, decimalJ_12, decimalT = self.countDecimal2()
        
        self.arrayJ_1 = (np.arange(self.j_1_inicial*decimalJ_1,decimalJ_1*self.j_1_final+self.passoJ_1*decimalJ_1, self.passoJ_1*decimalJ_1)/decimalJ_1).tolist()
        self.arrayJ_2 = (np.arange(self.j_2_inicial*decimalJ_2, decimalJ_2*self.j_2_final+self.passoJ_2*decimalJ_2, self.passoJ_2*decimalJ_2)/decimalJ_2).tolist()
        self.arrayBz_1 = (np.arange(self.bz_1_inicial*decimalBz_1, decimalBz_1*self.bz_1_final+self.passoBz_1*decimalBz_1, self.passoBz_1*decimalBz_1)/decimalBz_1).tolist()
        self.arrayBz_2 = (np.arange(self.bz_2_inicial*decimalBz_2, decimalBz_2*self.bz_2_final+self.passoBz_2*decimalBz_2, self.passoBz_2*decimalBz_2)/decimalBz_2).tolist()
        self.arrayJ_12 = (np.arange(self.j_12_inicial*decimalJ_12, decimalJ_12*self.j_12_final+self.passoJ_12*decimalJ_12, self.passoJ_12*decimalJ_12)/decimalJ_12).tolist()
        self.arrayT = (np.arange(self.tInicial*decimalT, decimalT*self.tFinal+self.passoT*decimalT, self.passoT*decimalT)/decimalT).tolist()
        self.elementos_iter = list(product(self.arrayJ_1, self.arrayJ_2, self.arrayBz_1, self.arrayBz_2,  self.arrayJ_12))
        #self.arrayJ_1 = np.array([])
        #self.arrayJ_2 = np.array([])
        #self.arrayBz_1 = np.array([])
        #self.arrayBz_2 = np.array([])
        #self.arrayJ_12 = np.array([])
        #self.arrayT = np.array([])
        self.dataSet = None
        self.model = None
        
     #Definição da equação da dinâmica de pontos quanticos versão mais completa.
    def hamiltoniana(self,j_1, j_2, bz_1, bz_2, j_12):
        #print("parametros hamiltoniana:", j_1, j_2, bz_1, bz_2, j_12)
        #input()
        return  0.5*(np.multiply(j_1, self.tensorProductSigZIdent) + np.multiply(j_2,self.tensorProductIdentSigZ) + 0.5*np.multiply(j_12,(self.tensorProductSigZSigZ - self.tensorProductSigZIdent - self.tensorProductIdentSigZ)) + np.multiply(bz_1,self.tensorProductSigXIdent) + np.multiply(bz_2,self.tensorProductIdentSigX))

    #Definição da equação da dinâmica de pontos quanticos versão mais completa.
    #@vectorize(target="cuda")
    def hamiltonianaNova(self, parametros):
        j_1 = parametros[0]
        j_2 = parametros[1]
        bz_1 = parametros[2]
        bz_2 = parametros[3] 
        j_12 = parametros[4]
        #print("parametros hamiltoniana:",parametros)
        #input()
        return  0.5*(np.multiply(j_1, self.tensorProductSigZIdent) + np.multiply(j_2,self.tensorProductIdentSigZ) + 0.5*np.multiply(j_12,(self.tensorProductSigZSigZ - self.tensorProductSigZIdent - self.tensorProductIdentSigZ)) + np.multiply(bz_1,self.tensorProductSigXIdent) + np.multiply(bz_2,self.tensorProductIdentSigX))

    
    #Definindo a função operador temporal.
    def u(self,t, h):
        eq1 = np.multiply(h,t)
        eq2 = np.multiply(eq1,(1j))
        eq3 = np.multiply(-1, eq2)
        #result = expm((np.matmul(np.matmul(h,t),(-1j))))
        result = expm(eq3)
        #print('result:', result)
        return result
    
    #Retorna o eigenvalues da multiplicação de ro com ro tempo reverso
    def get_eigvalues(self, ro, ro_tr):
        eigvalues, eigvectors = np.linalg.eig(np.matmul(ro,ro_tr))
        return eigvalues
    
    #Retorna a medida da concorrencia dado o ro.
    def concurrence(self, ro):
        ro_tr = self.ro_time_reversed(ro)
        eig_val = self.get_eigvalues(ro, ro_tr)
        eig_sqr_ord = np.sqrt(np.sort(eig_val)[::-1])
        eig_sum = eig_sqr_ord[0]
        for eig_sqrt in eig_sqr_ord[1:]:
            eig_sum -= eig_sqrt
        return np.maximum(0, eig_sum)
    
    #Definindo a função que calcula o Operador Densidade tempo-reverso 
    def ro_time_reversed(self, ro):
        tp_sigy_sigy = TensorProduct(self.sigmaY, self.sigmaY)
        ro_conj = np.conjugate(ro)
        return np.matmul(tp_sigy_sigy , np.matmul(ro_conj, tp_sigy_sigy))
    
    #Definindo a função operador densidade.
    def ro(self,t, h):
        u = self.u(t, h)
        #print("u:\n", u)
        #print("matrix:\n", np.matrix(u))
        #print("matrix dagger:\n", np.matrix(u).getH())
        #input()
        return np.dot(np.dot(u,self.ro0), np.array(np.matrix(u).getH()))


    #--------------------------------------------------
    #Observaveis:
    #Definindo a função O^(1)_x 
    def Ox1(self,ro):
        a = np.dot(self.tensorProductSigXIdent, ro)
        return np.trace(a)


    #Definindo a função O^(2)_x 
    def Ox2(self,ro):
        a = np.dot(self.tensorProductIdentSigX, ro)
        return np.trace(a)

    #--------------------------------------------------
    #Definindo a função O^(1)_y 
    def Oy1(self,ro):
        a = np.dot(self.tensorProductSigYIdent, ro)
        return np.trace(a)


    #Definindo a função O^(2)_y 
    def Oy2(self,ro):
        a = np.dot(self.tensorProductIdentSigY, ro)
        return np.trace(a)

    #--------------------------------------------------
    #Definindo a função O^(1)_z 
    def Oz1(self,ro):
        a = np.dot(self.tensorProductSigZIdent, ro)
        return np.trace(a)


    #Definindo a função O^(2)_z 
    def Oz2(self,ro):
        a = np.dot(self.tensorProductIdentSigZ, ro)
        return np.trace(a)
        
    def countDecimal(self):
        passos = [self.passoJ_1, self.passoJ_2, self.passoBz_1,self.passoBz_2, self.passoJ_12, self.passoT]
        decimals = np.array([])
        for passo in passos:
            decimals = np.append(decimals, 10**(-Decimal(str(passo)).as_tuple().exponent))
        return (decimals[0], decimals[1], decimals[2], decimals[3], decimals[4], decimals[5]) 
    

    #@vectorize(target="cuda")
    def criaFrame(self):
        #t0 = perf_counter()
        results = np.array([])
        t1 = t0 = perf_counter()
        j_12_len = len(self.arrayJ_12)
        for index,j12Dez in enumerate(self.arrayJ_12):
            #print("{:.1f}\n".format(index/j_12_len))
            #print("Total tempo gasto: ", t1 - t0)
            #t0 = perf_counter()
            j_12 = j12Dez
            for j1Dez in self.arrayJ_1:
                j_1 = j1Dez
                for j2Dez in self.arrayJ_2:
                    j_2 = j2Dez
                    for bz1Dez in self.arrayBz_1:
                        bz_1 = bz1Dez 
                        for bz2Dez in self.arrayBz_2:
                            bz_2 = bz2Dez
                            resultsOx = np.array([])
                            hvalor = self.hamiltoniana(j_1, j_2, bz_1, bz_2, j_12)
                            for tDez in self.arrayT:
                                t = tDez
                                rovalor = self.ro(t,hvalor)
                                ox1 = np.float32(self.Ox1(rovalor))
                                ox2 = np.float32(self.Ox2(rovalor))
                                oy1 = np.float32(self.Oy1(rovalor))
                                oy2 = np.float32(self.Oy2(rovalor))
                                oz1 = np.float32(self.Oz1(rovalor))
                                oz2 = np.float32(self.Oz2(rovalor))
                                resultsOx =  np.append(resultsOx,[ox1, ox2, oy1, oy2, oz1, oz2])
                            resultsOx = np.append([j_1, j_2, bz_1, bz_2, tDez], resultsOx)
                            results = np.append(results, resultsOx)
        t1 = perf_counter()
        #t1 = perf_counter()
        colunas = int((((((self.tFinal - self.tInicial)/self.passoT)+1)*6)+5))
        linhas = int(len(results)/colunas)
        
        print('colunas:', colunas)
        print("Total tempo gasto: ", t1 - t0)   
        print("results shape:", results.shape)
        print("Tamanho:", len(results))
        print('linhas:', linhas)
        return np.float32(results.reshape(linhas, colunas))
    
    #@vectorize(target="cuda")
    def calc_obs(self, hvalor):
        resultsOx = []
        for t in self.arrayT:
            rovalor = self.ro(t,hvalor)
            ox1 = np.float32(self.Ox1(rovalor))
            ox2 = np.float32(self.Ox2(rovalor))
            oy1 = np.float32(self.Oy1(rovalor))
            oy2 = np.float32(self.Oy2(rovalor))
            oz1 = np.float32(self.Oz1(rovalor))
            oz2 = np.float32(self.Oz2(rovalor))
            resultsOx.append([ox1, ox2, oy1, oy2, oz1, oz2])
        return resultsOx
    
    #@vectorize(target="cuda")
    def criaFrameNovo(self):
        #print(len(self.elementos_iter))
        t1 = t0 = perf_counter()
        
        reslts_hvalor =list(map(self.hamiltonianaNova, self.elementos_iter))
        resultsOxJ = list(map(self.calc_obs, list(reslts_hvalor)))
        
        t1 = perf_counter()
        colunas = int(((((self.tFinal - self.tInicial)/self.passoT)+1)*6))
        linhas = int(len(self.elementos_iter))
        print("elementos_iter_array: ", self.elementos_iter[:5])
        #print('colunas:', colunas)
        print("Total tempo gasto: ", t1 - t0)   
        #print("results shape:", results.shape)
        print("Tamanho:", len(resultsOxJ))
        print("resultados:", resultsOxJ[:5])
        #print('linhas:', linhas)
        
        #Compila o resultado com os elementos referentes a cada resultado
        result = np.hstack((np.reshape(np.array(self.elementos_iter), (linhas, 5)),np.reshape(np.array(resultsOxJ), (linhas, colunas))))
        print(result.shape)
        return result
    
    #dataframe
    def getNames(self):
        listO = [['ox1T' + str(tempo),'ox2T' + str(tempo), 'oy1T' + str(tempo), 'oy2T' + str(tempo), 'oz1T' + str(tempo),'oz2T' + str(tempo)] for tempo in self.arrayT]
        listOFlat = np.array([])
        for tempos in listO:
            listOFlat = np.append(listOFlat, np.array(tempos))
        return np.append(np.append(np.append(np.append(['j_1'],['j_2']),np.append(['bz_1'], ['bz_2'])), ['j_12_Target']) , listOFlat)
    
    def criaDataFrame(self):
        self.dataSet = pd.DataFrame(self.criaFrameNovo(), columns = self.getNames())
        self.saveDataFrame()
        return self.dataSet
    
    def loadDataFrame(self, path):
        self.dataSet = pd.read_csv(path, index_col = 0)
        return self.dataSet
    
    def grafico_acc_measures(self, path, accuracy, times):
        plt.plot(times, accuracy, color='blue', marker='o')
        plt.title('mse x tempos', fontsize=14)
        plt.xlabel('Quantidade de tempo utilizado', fontsize=14)
        plt.ylabel('média do erro ao quadrado', fontsize=14)
        plt.grid(True)
        plt.savefig("results/"+path[:-4]+"-msextempo"+".png")
        return
    
    def gen_acc_measures(self,path):
        dataSet = self.loadDataFrame(path)
        X = dataSet.loc[:,dataSet.columns.str.startswith('o')]
        y = dataSet.loc[:,dataSet.columns.str.startswith('j_12')]
        #print("x:", np.array(X.iloc[:,:6]).shape)
        #print("head x:", np.array(X.iloc[:,:6]))
        #print("y:", len(np.array(y)))
        #print("head y:", np.array(y).reshape(len(y),1))
        mse = []
        nb_measures = []
        
        for x_index in np.arange(start=6, stop=len(X.columns), step=6):
            mse_array = np.array([])
            for i in range(10):
                X_train, X_test, y_train, y_test = train_test_split(np.array(X.iloc[:,:x_index]).reshape(len(y), x_index), np.array(y).reshape(len(y),), test_size=0.3, random_state= 0)
                reg = ExtraTreesRegressor(n_estimators=100, random_state=0, n_jobs= -1).fit(X_train, y_train)
                #predict
                y_test_pred = reg.predict(X_test)
                mse_array = np.append(mse_array, mean_squared_error(y_test, y_test_pred))
            #quantidade de tempo
            qtd_tempo = int(x_index/6)

            nb_measures.append(qtd_tempo)
            mse.append(np.mean(mse_array))
        
        self.grafico_acc_measures(path, mse, nb_measures)
        return
        
    def saveDataFrame(self):
        if self.dataSet is None:
             self.criaDataFrame()
        self.dataSet.to_csv(path_or_buf="./data/TabelasNovas/" + self.name + ".csv")
        return
    
    def saveDataFrameY(self, datasetY):
        datasetY.to_csv(path_or_buf="./data/TabelasNovas/Y" +"yRealyPred -"+ self.name + ".csv")
        return
    
    def save_Y(self, yReal, yPred, lenght):
        print("shape:", np.array(yReal).reshape((lenght,-1)).shape)
        yReal = np.array(yReal).reshape((lenght,-1))
        yPred = np.array(yPred).reshape((lenght,-1))
        datasetY = pd.DataFrame(np.hstack([yReal, yPred]), columns = ["yReal","yPred"])
        self.saveDataFrameY(datasetY)
        return datasetY
    
    def criaFrameGraficosNovo(self):
        t0 = perf_counter()
        results = np.array([])
        ox1 = np.array([])
        ox2 = np.array([])
        oy1 = np.array([])
        oy2 = np.array([])
        oz1 = np.array([])
        oz2 = np.array([])
        tempos = np.array([])
        for j_12 in self.arrayJ_12:
            for j_1 in self.arrayJ_1:
                for j_2 in self.arrayJ_2:
                    for bz_1 in self.arrayBz_1:
                        for bz_2 in self.arrayBz_2:
                            resultsOx = np.array([])
                            hvalor = self.hamiltoniana(j_1, j_2, bz_1, bz_2, j_12)
                            for t in self.arrayT:
                                rovalor = self.ro(t,hvalor)
                                ox1 = np.float32(np.append(ox1, self.Ox1(rovalor)))
                                ox2 = np.float32(np.append(ox2, self.Ox2(rovalor)))
                                oy1 = np.float32(np.append(oy1, self.Oy1(rovalor)))
                                oy2 = np.float32(np.append(oy2, self.Oy2(rovalor)))
                                oz1 = np.float32(np.append(oz1, self.Oz1(rovalor)))
                                oz2 = np.float32(np.append(oz2, self.Oz2(rovalor)))
                                tempos = np.append(tempos, t)

        t1 = perf_counter()
        
        print('ox1 shape:', ox1.shape)
        print('ox2 shape:', ox2.shape)
        print('oy1 shape:', oy1.shape)
        print('oy2 shape:', oy2.shape)
        print('oz1 shape:', oz1.shape)
        print('oz2 shape:', oz2.shape)
        print('tempos shape:', tempos.shape)
        print("Total tempo gasto: ", t1 - t0)
        return pd.DataFrame(ox1, columns = ['ox1']), pd.DataFrame(ox2, columns = ['ox2']), pd.DataFrame(oy1, columns = ['oy1']), pd.DataFrame(oy2, columns = ['oy2']), pd.DataFrame(oz1, columns = ['oz1']), pd.DataFrame(oz2, columns = ['oz2']), pd.DataFrame(tempos, columns = ['tempo'])

    def criaGraficos(self, dataFrame, tempos,nome):
        fig, ax = plt.subplots()
        ax.plot(tempos, dataFrame)
        ax.set(xlabel='tempo', ylabel = dataFrame.columns, title = dataFrame.columns)
        ax.grid()
        fig.savefig(nome+".png")
        #plt.show()
        return

    ###Teste Aleatório
    def random_float(self,low, high):
         return random.random()*(high-low) + low

    def random_samples(self,low, high, k):
        samples = []
        for number_samples in range(k):
            samples.append(self.random_float(low,high))
        return samples
    
    ### modelo
    def set_model(self, df = ""):
        if df == "":
            df = self.dataSet
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,5:], df.iloc[:,4], test_size=0.3, random_state= 0)
        self.model = ExtraTreesRegressor(n_estimators=100, random_state=0, n_jobs= -1).fit(X_train, y_train)
        return self.model, X_train, X_test, y_train, y_test

    def make_report(self, name, y_real, y_pred):
        relative_error_1 = 0
        relative_error_2 = 0

        mae = mean_absolute_error(y_real, y_pred)
        mse = mean_squared_error(y_real, y_pred)
        r2 = r2_score(y_real,y_pred)

        for y_p, y_t in zip(y_pred, y_real):
            relative_error_1 += np.abs(1-y_p)/y_t
            relative_error_2 += np.abs((y_t-y_p)/y_t)

        with open("Resultados.txt", "a+") as text_file:
            text_file.write("===="*5+"\n")
            text_file.write(name)
            text_file.write("\n%s"%(self.name_comp))
            text_file.write("\nMédia do erro absoluto: %f \nMédia quadrada do erro: %f \nR2: %f\nrelative_error_1: %f \nrelative_error_2: %f" % (mae, mse, r2, relative_error_1, relative_error_2))

        graphics.plotGraph(y_real, y_pred, "Extra Trees Regressor"+name,self.name, mae, mse, r2)

        return

    def make_speed(self,t0,t1, subname = "", name = ""):
        if name == "":
            name = self.name_comp
        with open("Speed.txt", "a+") as text_file:
            text_file.write("===="*5)
            text_file.write(subname+"\n")
            text_file.write("\n%s"%(name))
            text_file.write("Tempo: %f \n" % (t1-t0))
        return

    def test_aleatorio(self, k):
        arrayJ_1 = self.random_samples(self.j_1_inicial, self.j_2_final, k)
        arrayJ_2 = self.random_samples(self.j_2_inicial, self.j_2_final, k)
        arrayBz_1 = self.random_samples(self.bz_1_inicial, self.bz_1_final, k)
        arrayBz_2 = self.random_samples(self.bz_2_inicial, self.bz_2_final, k)
        arrayJ_12 = self.random_samples(self.j_12_inicial, self.j_12_final, k)

        self.elementos_iter = list(product(arrayJ_1, arrayJ_2, arrayBz_1, arrayBz_2,  arrayJ_12))

        df = self.criaDataFrame()
        X_val = df.iloc[:,5:]
        Y_val = df.iloc[:,4]
        y_val_pred = self.model.predict(X_val)

        make_results("teste aleatorio com K:"%k, y_val, y_val_pred)

        return

    def make_results(self, k = 10000):
        t0 = perf_counter()
        self.criaDataFrame()
        model, X_train, X_test, y_train, y_test = self.set_model()

        y_train_pred = model.predict(X_train)
        self.make_report("Treino", y_train, y_train_pred)

        y_test_pred = model.predict(X_test)
        self.make_report("Teste", y_test, y_test_pred)
        t1 = perf_counter()
        self.make_speed(t0 = t0, t1 = t1)

        t0 = perf_counter()
        self.test_aleatorio(k)
        t1 = perf_counter()
        self.make_speed(t0 = t0, t1 = t1)

        return