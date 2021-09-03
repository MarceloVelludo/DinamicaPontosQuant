#Para construir a tabela(dataframe) que será usada para treinar o modelo vamos usar o Pandas.
import numpy as np                                                                                                                  
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pycaret.utils import enable_colab
from pycaret.regression import * 
from sympy.physics.quantum import TensorProduct
from time import perf_counter 
from scipy.linalg import expm
from cmath import  *
from decimal import *
#from math import *

#A classe DinamicaPontosQuanticos calcula a dinamica no intervalo desejado e cria a tabela com os resultados.
class DinamicaPontosQuanticos:
  
    def __init__(self, aInicial=0, aFinal=1, passoA = 0.1, bInicial=0
                 , bFinal=1, passoB = 0.1, jInicial=0, jFinal=10, passoJ = 0.1 
                 , tInicial=5, tFinal=25, passoT=5):
        self.aInicial = aInicial
        self.aFinal = aFinal
        self.bInicial = bInicial
        self.bFinal = bFinal
        self.jInicial = jInicial
        self.jFinal = jFinal
        self.tInicial = tInicial
        self.tFinal = tFinal
        self.passoA = passoA
        self.passoB = passoB
        self.passoJ = passoJ
        self.passoT = passoT
        
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
        
        self.arrayA = np.array([])
        self.arrayB = np.array([])
        self.arrayJ = np.array([])
        self.arrayT = np.array([])
        self.dataSet = None

    #Definição da equação da dinâmica de pontos quanticos.
    def hamiltoniana(self,a,b,j):
        #print('a:\n', a)
        #print('b:\n', b)
        #print('j:\n', j)
        #print('resultado:\n', np.matmul(a,self.tensorProductIdentSigX) + np.matmul(b,self.tensorProductSigXIdent) + self.tensorProductSigZIdentSoma  + np.matmul(j,self.tensorProductSigZSigZ))
        #input()
        return  np.multiply(a,self.tensorProductIdentSigX) + np.multiply(b,self.tensorProductSigXIdent) + self.tensorProductSigZIdentSoma  + np.multiply(j,self.tensorProductSigZSigZ)
    
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
        return np.matmul(np.matmul(u,self.ro0), np.array(np.matrix(u).getH()))


    #--------------------------------------------------
    #Observaveis:
    #Definindo a função O^(1)_x 
    def Ox1(self,ro):
        a = np.matmul(self.tensorProductSigXIdent, ro)
        return np.trace(a)


    #Definindo a função O^(2)_x 
    def Ox2(self,ro):
        a = np.matmul(self.tensorProductIdentSigX, ro)
        return np.trace(a)

    #--------------------------------------------------
    #Definindo a função O^(1)_y 
    def Oy1(self,ro):
        a = np.matmul(self.tensorProductSigYIdent, ro)
        return np.trace(a)


    #Definindo a função O^(2)_y 
    def Oy2(self,ro):
        a = np.matmul(self.tensorProductIdentSigY, ro)
        return np.trace(a)

    #--------------------------------------------------
    #Definindo a função O^(1)_z 
    def Oz1(self,ro):
        a = np.matmul(self.tensorProductSigZIdent, ro)
        return np.trace(a)


    #Definindo a função O^(2)_z 
    def Oz2(self,ro):
        a = np.matmul(self.tensorProductIdentSigZ, ro)
        return np.trace(a)
        
    def countDecimal(self):
        passos = [self.passoA, self.passoB, self.passoJ, self.passoT]
        decimals = np.array([])
        for passo in passos:
            decimals = np.append(decimals, 10**(-Decimal(str(passo)).as_tuple().exponent))
        return (decimals[0], decimals[1], decimals[2], decimals[3]) 
    


    def criaFrame(self):
        t0 = perf_counter()
        results = np.array([])
        decimalA, decimalB, decimalJ, decimalT = self.countDecimal()
        #print("inicial:", self.jInicial*decimalJ)
        #print("final:", decimalJ*self.jFinal+self.passoJ*decimalJ)
        #print("passo:", self.passoJ*decimalJ)
        #print('self.passo:', self.passoJ)
        #print('decimalA:', decimalA)
        #print('decimalB:', decimalB)
        #print('decimalJ:', decimalJ)
        #print('decimalT:', decimalT)
        self.arrayA = np.arange(self.aInicial*decimalA, decimalA*self.aFinal+self.passoA*decimalA, self.passoA*decimalA)
        self.arrayB = np.arange(self.bInicial*decimalB, decimalB*self.bFinal+self.passoB*decimalB, self.passoB*decimalB)
        self.arrayJ = np.arange(self.jInicial*decimalJ, decimalJ*self.jFinal+self.passoJ*decimalJ, self.passoJ*decimalJ)
        self.arrayT = np.arange(self.tInicial*decimalT, decimalT*self.tFinal+self.passoT*decimalT, self.passoT*decimalT)
        #print("arrayJ:", arrayJ)
        #print("arrayA:", arrayA)
        #print("arrayB:", arrayB)
        #print("arrayT:", arrayT)
        
        for jDez in self.arrayJ:
            j = jDez/decimalJ
            for aDez in self.arrayA:
                a = aDez/decimalA 
                for bDez in self.arrayB:
                    b = bDez/decimalB
                    resultsOx = np.array([])
                    hvalor = self.hamiltoniana(a, b, j)
                    for tDez in self.arrayT:
                        t = tDez/decimalT
                        rovalor = self.ro(t,hvalor)
                        ox1 = np.float32(self.Ox1(rovalor))
                        ox2 = np.float32(self.Ox2(rovalor))
                        oy1 = np.float32(self.Oy1(rovalor))
                        oy2 = np.float32(self.Oy2(rovalor))
                        oz1 = np.float32(self.Oz1(rovalor))
                        oz2 = np.float32(self.Oz2(rovalor))
                        resultsOx =  np.append(resultsOx,[ox1, ox2, oy1, oy2, oz1, oz2])

                    resultsOxJ = np.append(np.append(np.append(a,b), j),resultsOx)
                    results = np.append(results, resultsOxJ)

        t1 = perf_counter()
        colunas = int((((((self.tFinal - self.tInicial)/self.passoT)+1)*6)+3))
        linhas = int(len(results)/colunas)
        
        print('colunas:', colunas)
        print("Total tempo gasto: ", t1 - t0)   
        print("results shape:", results.shape)
        print("Tamanho:", len(results))
        print('linhas:', linhas)
        return np.float32(results.reshape(linhas, colunas))
    
    #dataframe
    def getNames(self):
        listO = [['ox1T' + str(tempo),'ox2T' + str(tempo), 'oy1T' + str(tempo), 'oy2T' + str(tempo), 'oz1T' + str(tempo),'oz2T' + str(tempo)] for tempo in self.arrayT]
        listOFlat = np.array([])
        for tempos in listO:
            listOFlat = np.append(listOFlat, np.array(tempos))
        return np.append(np.append(np.append(['a'],['b']), ['jTarget']) , listOFlat)
    
    def criaDataFrame(self):
        self.dataSet = pd.DataFrame(self.criaFrame(), columns = self.getNames())
        return self.dataSet
    
    def criaFrameGraficos(self):
        t0 = perf_counter()
        results = np.array([])
        decimalA, decimalB, decimalJ, decimalT = self.countDecimal()
        print("inicial:", self.jInicial*decimalJ)
        print("final:", decimalJ*self.jFinal+self.passoJ*decimalJ)
        print("passo:", self.passoJ*decimalJ)
        print('self.passo:', self.passoJ)
        print('decimalA:', decimalA)
        print('decimalB:', decimalB)
        print('decimalJ:', decimalJ)
        print('decimalT:', decimalT)
        arrayA = np.arange(self.aInicial*decimalA, decimalA*self.aFinal+self.passoA*decimalA, self.passoA*decimalA)
        arrayB = np.arange(self.bInicial*decimalB, decimalB*self.bFinal+self.passoB*decimalB, self.passoB*decimalB)
        arrayJ = np.arange(self.jInicial*decimalJ, decimalJ*self.jFinal+self.passoJ*decimalJ, self.passoJ*decimalJ)
        arrayT = np.arange(self.tInicial*decimalT, decimalT*self.tFinal+self.passoT*decimalT, self.passoT*decimalT)
        ox1 = np.array([])
        ox2 = np.array([])
        oy1 = np.array([])
        oy2 = np.array([])
        oz1 = np.array([])
        oz2 = np.array([])
        tempos = np.array([])
        print("arrayJ:", arrayJ/decimalJ)
        print("arrayA:", arrayA/decimalA)
        print("arrayB:", arrayB/decimalB)
        print("arrayT:", arrayT/decimalT)
        for jDez in arrayJ:
            j = jDez/decimalJ
            for aDez in arrayA:
                a = aDez/decimalA 
                for bDez in arrayB:
                    b = bDez/decimalB
                    resultsOx = np.array([])
                    hvalor = self.hamiltoniana(a, b, j)
                    for tDez in arrayT:
                        t = tDez/10
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
    
    def criaGraficos(self, dataFrame, tempos):
        fig, ax = plt.subplots()
        ax.plot(tempos, dataFrame)
        ax.set(xlabel='tempo', ylabel = dataFrame.columns, title = dataFrame.columns)
        ax.grid()
        #fig.savefig("test.png")
        plt.show()
    
    def saveDataFrame(self):
        if self.dataSet is None:
             self.criaDataFrame()
        nome = str("["+str(self.aInicial)+":"+str(self.aFinal)+"]"+"["+str(self.bInicial)+":"+str(self.bFinal)+"]"+"["+str(self.jInicial)+":"+str(self.jFinal)+"]"+"["+str(self.tInicial)+":"+str(self.tFinal)+"]")
        self.dataSet.to_csv(path_or_buf="./data/" + nome + ".csv")
        return