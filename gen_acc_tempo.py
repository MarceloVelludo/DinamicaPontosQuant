import pandas as pd
import DPQNova
from os import listdir
from os.path import isfile, join

#Path directory.
path = "data/TabelasNovas/"
#Capture files in the path directory
onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and "Y" not in str(f)]



print("\nlista contém %i caminhos"%len(onlyfiles))
print(onlyfiles)
for file_path in onlyfiles:
    try:
        dpq = DPQNova.DinamicaPontosQuanticos()
        dpq.gen_acc_measures(file_path)
    except:
        print("path:%s, mal sucedido!"%file_path)