import numpy as np
import pandas as pd
import neurolab as nl
import pylab as pl
import os

# Estraggo le informazioni necessarie per restituire un corretto output dalla rete
datasetInfo = pd.read_csv("C:\\Users\\iacop\\Documents\\projects\\Python\\Esercitazione\\example1\\results\\datasetInfo.csv")
attr1Max = datasetInfo["max"][0]
attr1Min = datasetInfo["min"][0]
attr2Max = datasetInfo["max"][1]
attr2Min = datasetInfo["min"][1]
attr3Max = datasetInfo["max"][2]
attr3Min = datasetInfo["min"][2]
attr4Max = datasetInfo["max"][3]
attr4Min = datasetInfo["min"][3]
resultMax = datasetInfo["max"][4]
resultMin = datasetInfo["min"][4]

# Estraggo tutte le reti neurali presenti nella cartella
files = os.listdir("results")
netFiles = []

for file in files:
    f = file.split(".")
    if (len(f) > 1):
        if f[1] == "net":
            netFiles.append(f[0])

# Elenco a video tutte le reti neurali disponibili nella cartella
print("There are " + str(len(netFiles)) + " neural network ready to load:")

for index, net in enumerate(netFiles):
    print(str(index+1) + ") " + net)

# Chiedo all'utente di specificare una rete tra quelle disponibili
print("Witch network to load?")
selectedNetwork = input()

try:

    if(int(selectedNetwork) > len(netFiles)):
        raise

    else:

        # Carico la rete selezionata
        net = nl.load("results/" + netFiles[int(selectedNetwork)-1] + ".net")

        while(True):
            # Chiedo all'utente di inserire i quattro attributi di input
            print("Put four attributes:")
            print("attr1:")
            attr1 = (float(input()) - attr1Min) / (attr1Max - attr1Min)
            print("attr2:")
            attr2 = (float(input()) - attr2Min) / (attr2Max - attr2Min)
            print("attr3:")
            attr3 = (float(input()) - attr3Min) / (attr3Max - attr3Min)
            print("attr4:")
            attr4 = (float(input()) - attr4Min) / (attr4Max - attr4Min)

            # Calcolo l'output relativo ai quattro attributi e lo mostro a video
            res = net.sim([[attr1, attr2, attr3, attr4]])[0][0]
            res = res*resultMax - res*resultMin + resultMin
            print("network output: " + str(res))

except:
    
    print("Invalid input!")
    exit