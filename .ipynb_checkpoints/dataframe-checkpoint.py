#Para construir a tabela(dataframe) que será usada para treinar o modelo vamos usar o Pandas.
import numpy as np                                     
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools
from itertools import product
from numba import jit, vectorize, cuda 
from sklearn.model_selection import train_test_split
from sympy.physics.quantum import TensorProduct
from time import perf_counter 
from scipy.linalg import expm
from cmath import  *
from decimal import *
from multiprocessing import Pool

j_1_inicial=0
j_1_final=1
passoJ_1=0.5
j_2_inicial=0
j_2_final=1
passoJ_2 =0.5
bz_1_inicial=0
bz_1_final=1
passoBz_1 = 0.5
bz_2_inicial=0
bz_2_final=1
passoBz_2 = 0.5
j_12_inicial=0
j_12_final=10
passoJ_12 = 0.1
tInicial=5
tFinal=25
passoT=5
#roInicial
#UpUp
#ro0 = np.array([[1,0,0,0], [0,0,0,0], [0,0,0,0],[0,0,0,0]])
ro0 = np.array([[0.25,0.25,0.25,0.25], [0.25,0.25,0.25,0.25], [0.25,0.25,0.25,0.25],[0.25,0.25,0.25,0.25]])
#Criando as matrizes de Pauli-X, Pauli-Y, e Pauli-Z.
sigmaX = np.array([[0, 1], [1, 0]])
sigmaY = np.array([[0, -1j], [1j, 0]])
sigmaZ = np.array([[1, 0], [0, -1]])

#Matriz identidade.
ident = np.identity(2)

#Algumas constantes que são usadas diversas vezes para os calculos.
tensorProductIdentSigX = TensorProduct(ident, sigmaX)
tensorProductSigXIdent = TensorProduct(sigmaX, ident)
tensorProductIdentSigY = TensorProduct(ident, sigmaY)
tensorProductSigYIdent = TensorProduct(sigmaY, ident)
tensorProductIdentSigZ = TensorProduct(ident, sigmaZ)
tensorProductSigZIdent = TensorProduct(sigmaZ, ident)
tensorProductSigZIdentSoma = TensorProduct(sigmaZ, ident) + TensorProduct(ident, sigmaZ)
tensorProductSigZSigZ = TensorProduct(sigmaZ, sigmaZ)

def countDecimal2():
    passos = [passoJ_1, passoJ_2, passoBz_1,passoBz_2, passoJ_12, passoT]
    decimals = np.array([])
    for passo in passos:
        decimals = np.append(decimals, 10**(-Decimal(str(passo)).as_tuple().exponent))

    return (decimals[0], decimals[1], decimals[2], decimals[3], decimals[4], decimals[5]) 

decimalJ_1, decimalJ_2, decimalBz_1, decimalBz_2, decimalJ_12, decimalT = countDecimal2()
arrayJ_1 = (np.arange(j_1_inicial*decimalJ_1,decimalJ_1*j_1_final+passoJ_1*decimalJ_1, passoJ_1*decimalJ_1)/decimalJ_1).tolist()
arrayJ_2 = (np.arange(j_2_inicial*decimalJ_2, decimalJ_2*j_2_final+passoJ_2*decimalJ_2, passoJ_2*decimalJ_2)/decimalJ_2).tolist()
arrayBz_1 = (np.arange(bz_1_inicial*decimalBz_1, decimalBz_1*bz_1_final+passoBz_1*decimalBz_1, passoBz_1*decimalBz_1)/decimalBz_1).tolist()
arrayBz_2 = (np.arange(bz_2_inicial*decimalBz_2, decimalBz_2*bz_2_final+passoBz_2*decimalBz_2, passoBz_2*decimalBz_2)/decimalBz_2).tolist()
arrayJ_12 = (np.arange(j_12_inicial*decimalJ_12, decimalJ_12*j_12_final+passoJ_12*decimalJ_12, passoJ_12*decimalJ_12)/decimalJ_12).tolist()
arrayT = (np.arange(tInicial*decimalT, decimalT*tFinal+passoT*decimalT, passoT*decimalT)/decimalT).tolist()
elementos_iter = list(product(arrayJ_1, arrayJ_2, arrayBz_1, arrayBz_2,  arrayJ_12))
#arrayJ_1 = np.array([])
#arrayJ_2 = np.array([])
#arrayBz_1 = np.array([])
#arrayBz_2 = np.array([])
#arrayJ_12 = np.array([])
#arrayT = np.array([])
dataSet = None

 #Definição da equação da dinâmica de pontos quanticos versão mais completa.
def hamiltoniana(j_1, j_2, bz_1, bz_2, j_12):
    #print("parametros hamiltoniana:", j_1, j_2, bz_1, bz_2, j_12)
    #input()
    return  0.5*(np.multiply(j_1, tensorProductSigZIdent) + np.multiply(j_2,tensorProductIdentSigZ) + 0.5*np.multiply(j_12,(tensorProductSigZSigZ - tensorProductSigZIdent - tensorProductIdentSigZ)) + np.multiply(bz_1,tensorProductSigXIdent) + np.multiply(bz_2,tensorProductIdentSigX))

#Definição da equação da dinâmica de pontos quanticos versão mais completa.
#@vectorize(target="cuda")
def hamiltonianaNova( parametros):
    j_1 = parametros[0]
    j_2 = parametros[1]
    bz_1 = parametros[2]
    bz_2 = parametros[3] 
    j_12 = parametros[4]
    #print("parametros hamiltoniana:",parametros)
    #input()
    return  0.5*(np.multiply(j_1, tensorProductSigZIdent) + np.multiply(j_2,tensorProductIdentSigZ) + 0.5*np.multiply(j_12,(tensorProductSigZSigZ - tensorProductSigZIdent - tensorProductIdentSigZ)) + np.multiply(bz_1,tensorProductSigXIdent) + np.multiply(bz_2,tensorProductIdentSigX))


#Definindo a função operador temporal.
def u_op(t, h):
    eq1 = np.multiply(h,t)
    eq2 = np.multiply(eq1,(1j))
    eq3 = np.multiply(-1, eq2)
    #result = expm((np.matmul(np.matmul(h,t),(-1j))))
    result = expm(eq3)
    #print('result:', result)
    return result

#Retorna o eigenvalues da multiplicação de ro com ro tempo reverso
def get_eigvalues( ro, ro_tr):
    eigvalues, eigvectors = np.linalg.eig(np.matmul(ro,ro_tr))
    return eigvalues

#Retorna a medida da concorrencia dado o ro.
def concurrence( ro):
    ro_tr = ro_time_reversed(ro)
    eig_val = get_eigvalues(ro, ro_tr)
    eig_sqr_ord = np.sqrt(np.sort(eig_val)[::-1])
    eig_sum = eig_sqr_ord[0]
    for eig_sqrt in eig_sqr_ord[1:]:
        eig_sum -= eig_sqrt
    return np.maximum(0, eig_sum)

#Definindo a função que calcula o Operador Densidade tempo-reverso 
def ro_time_reversed( ro):
    tp_sigy_sigy = TensorProduct(sigmaY, sigmaY)
    ro_conj = np.conjugate(ro)
    return np.matmul(tp_sigy_sigy , np.matmul(ro_conj, tp_sigy_sigy))

#Definindo a função operador densidade.
def ro(t, h):
    u = u_op(t, h)
    return np.matmul(np.matmul(u,ro0), np.array(np.matrix(u).getH()))


#--------------------------------------------------
#Observaveis:
#Definindo a função O^(1)_x 
def Ox1(ro):
    a = np.matmul(tensorProductSigXIdent, ro)
    return np.trace(a)


#Definindo a função O^(2)_x 
def Ox2(ro):
    a = np.matmul(tensorProductIdentSigX, ro)
    return np.trace(a)

#--------------------------------------------------
#Definindo a função O^(1)_y 
def Oy1(ro):
    a = np.matmul(tensorProductSigYIdent, ro)
    return np.trace(a)


#Definindo a função O^(2)_y 
def Oy2(ro):
    a = np.matmul(tensorProductIdentSigY, ro)
    return np.trace(a)

#--------------------------------------------------
#Definindo a função O^(1)_z 
def Oz1(ro):
    a = np.matmul(tensorProductSigZIdent, ro)
    return np.trace(a)


#Definindo a função O^(2)_z 
def Oz2(ro):
    a = np.matmul(tensorProductIdentSigZ, ro)
    return np.trace(a)

def countDecimal():
    passos = [passoJ_1, passoJ_2, passoBz_1,passoBz_2, passoJ_12, passoT]
    decimals = np.array([])
    for passo in passos:
        decimals = np.append(decimals, 10**(-Decimal(str(passo)).as_tuple().exponent))
    return (decimals[0], decimals[1], decimals[2], decimals[3], decimals[4], decimals[5]) 


#@jit
def criaFrame():
    #t0 = perf_counter()
    results = np.array([])
    #t1 = t0 = perf_counter()
    j_12_len = len(arrayJ_12)
    for index,j12Dez in enumerate(arrayJ_12):
        #print("{:.1f}\n".format(index/j_12_len))
        #print("Total tempo gasto: ", t1 - t0)
        #t0 = perf_counter()
        j_12 = j12Dez
        for j1Dez in arrayJ_1:
            j_1 = j1Dez
            for j2Dez in arrayJ_2:
                j_2 = j2Dez
                for bz1Dez in arrayBz_1:
                    bz_1 = bz1Dez 
                    for bz2Dez in arrayBz_2:
                        bz_2 = bz2Dez
                        resultsOx = np.array([])
                        hvalor = hamiltoniana(j_1, j_2, bz_1, bz_2, j_12)
                        for tDez in arrayT:
                            t = tDez
                            rovalor = ro(t,hvalor)
                            ox1 = np.float32(Ox1(rovalor))
                            ox2 = np.float32(Ox2(rovalor))
                            oy1 = np.float32(Oy1(rovalor))
                            oy2 = np.float32(Oy2(rovalor))
                            oz1 = np.float32(Oz1(rovalor))
                            oz2 = np.float32(Oz2(rovalor))
                            resultsOx =  np.append(resultsOx,[ox1, ox2, oy1, oy2, oz1, oz2])
                        resultsOx = np.append([j_1, j_2, bz_1, bz_2, tDez], resultsOx)
                        results = np.append(results, resultsOx)
    #t1 = perf_counter()
    #t1 = perf_counter()
    colunas = int((((((tFinal - tInicial)/passoT)+1)*6)+5))
    linhas = int(len(results)/colunas)

    print('colunas:', colunas)
    #print("Total tempo gasto: ", t1 - t0)   
    print("results shape:", results.shape)
    print("Tamanho:", len(results))
    print('linhas:', linhas)
    return np.float32(results.reshape(linhas, colunas))
#@vectorize(target="cuda")
def calc_obs( hvalor):
    resultsOx = []
    for t in arrayT:
        rovalor = ro(t,hvalor)
        ox1 = np.float32(Ox1(rovalor))
        ox2 = np.float32(Ox2(rovalor))
        oy1 = np.float32(Oy1(rovalor))
        oy2 = np.float32(Oy2(rovalor))
        oz1 = np.float32(Oz1(rovalor))
        oz2 = np.float32(Oz2(rovalor))
        resultsOx.append([ox1, ox2, oy1, oy2, oz1, oz2])
    return resultsOx
#@vectorize(target="cuda")
def criaFrameNovo():

    #results = []
    print(len(elementos_iter))
    t1 = t0 = perf_counter()
    #with Pool(8) as pool:
    #print("elementos_iter:",elementos_iter[:5])
    reslts_hvalor =list(map(hamiltonianaNova, elementos_iter))
    #print(reslts_hvalor)
    #print("tipo reslts_havlor:{}, tipo calc_obs:{}".format(type(reslts_hvalor), type(calc_obs)))
    #with Pool(16) as pool:
    resultsOxJ = list(map(calc_obs, list(reslts_hvalor)))
    #resultsOxJ.append([elementos[0],elementos[1],elementos[2], elementos[3], elementos[4],resultsOx])
    #results.append(resultsOxJ)

    t1 = perf_counter()
    #colunas = int((((((tFinal - tInicial)/passoT)+1)*6)+5))
    #linhas = int(len(results)/colunas)

    #print('colunas:', colunas)
    print("Total tempo gasto: ", t1 - t0)   
    #print("results shape:", results.shape)
    print("Tamanho:", len(resultsOxJ))
    #print('linhas:', linhas)
    return resultsOxJ

@cuda.jit
def criaFrameNovoGPU(elementos_iter_gpu, results_gpu):
    # Compute flattened index inside the array
    pos = pos = cuda.grid(1)
    if pos < elementos_iter_gpu.size:  # Check array boundaries

        reslts_hvalor = np.array(hamiltonianaNova(elementos_iter_gpu[pos]))
        results_gpu[pos] = np.array(map(calc_obs, reslts_hvalor))
                                    
def better_call_gpu():
    elementos_iter = np.asarray(product(arrayJ_1, arrayJ_2, arrayBz_1, arrayBz_2,  arrayJ_12))
    elementos_iter_gpu = cuda.to_device(elementos_iter)
    results = np.zeros(elementos_iter.size)
    results_gpu = cuda.to_device(results)
    #arrayT_gpu = cuda.to_device(arrayT)
    #ro0_gpu = cuda.to_device(ro0)
    #tensorProductIdentSigX_gpu = cuda.to_device(tensorProductIdentSigX)
    #tensorProductSigXIdent_gpu = cuda.to_device(tensorProductSigXIdent)
    #tensorProductIdentSigY_gpu = cuda.to_device(tensorProductIdentSigY)
    #tensorProductSigYIdent_gpu = cuda.to_device(tensorProductSigYIdent)
    #tensorProductIdentSigZ_gpu = cuda.to_device(tensorProductIdentSigZ)
    #tensorProductSigZIdent_gpu = cuda.to_device(tensorProductSigZIdent)
    #tensorProductSigZIdentSoma_gpu = cuda.to_device(tensorProductSigZIdentSoma)
    #tensorProductSigZSigZ_gpu = cuda.to_device(tensorProductSigZSigZ)
    # Set the number of threads in a block
    threadsperblock = 32 
    # Calculate the number of thread blocks in the grid
    blockspergrid = (elementos_iter_gpu.size + (threadsperblock - 1)) // threadsperblock
    
    #resultsOxJ = list([])
    # Now start the kernel
    #my_kernel[blockspergrid, threadsperblock](elementos_iter_gpu, arrayT, ro0, tensorProductIdentSigX, tensorProductSigXIdent, tensorProductIdentSigY, tensorProductSigYIdent, tensorProductIdentSigZ , tensorProductSigZIdent, tensorProductSigZIdentSoma, tensorProductSigZSigZ)
    criaFrameNovoGPU[blockspergrid, threadsperblock](elementos_iter_gpu, results_gpu) 
    results_gpu = results_gpu.copy_to_host()