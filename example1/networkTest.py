import numpy as np
import pandas as pd
import neurolab as nl
import pylab as pl
import os

# Estraggo tutte le reti neurali presenti nella cartella
files = os.listdir("results")
netFiles = []

for file in files:
    f = file.split(".")
    if (len(f) > 1):
        if f[1] == "net":
            netFiles.append(f[0])

# Estraggo testSet e trainingSet utilizzati per l'addestramento
testSet = pd.read_csv("results/testSet.csv")
testSetInputs = testSet[["attr1", "attr2", "attr3", "attr4"]]
testSetTargets = testSet[["result"]].rename(columns={"result" : "targets"})

trainingSet = pd.read_csv("results/trainingSet.csv")
trainingSetInputs = trainingSet[["attr1", "attr2", "attr3", "attr4"]]
trainingSetTargets = trainingSet[["result"]].rename(columns={"result" : "targets"})

# Creo una struttura dati pandas.DataFrame per inserire tutte le informazioni salienti
netAnalysisReport = []

# Scorro tutte le reti neurali presenti nella cartella
for neuralNet in netFiles:

    # Inserisco un nuovo elemento nella struttura dati addetta all'analisi finale
    netAnalysisReport.append([int(neuralNet.split("Neurons")[0]),0,0])

    # Carico l'n-esima rete neurale
    net = nl.load("results/" + neuralNet + ".net")

    # Eseguo una simulazione sul testSet
    testSetResults = pd.DataFrame(net.sim(testSetInputs)).rename(columns={0 : "results"})

    result = pd.concat([testSetTargets, testSetResults], axis=1)

    # Calcolo l'errore quadratico sulle singole tuple tra il risultato ottenuto e il risultato atteso
    squaredError = []

    for i in result.index:
        squaredError.append(pow((result["targets"][i] - result["results"][i]),2))

    netAnalysisReport[-1][1] = sum(squaredError)/len(squaredError)

    result = pd.concat([result, pd.DataFrame(squaredError).rename(columns={0 : "squaredErr"})], axis=1)

    # Stampo il risultato in csv
    result.to_csv("results/" + str(neuralNet) + "_testOnTestSet.csv", index=False)

    # Eseguo una simulazione sul trainingSet
    trainingSetResults = pd.DataFrame(net.sim(trainingSetInputs)).rename(columns={0 : "results"})

    result = pd.concat([trainingSetTargets, trainingSetResults], axis=1)

    # Calcolo l'errore quadratico sulle singole tuple tra il risultato ottenuto e il risultato atteso
    squaredError = []

    for i in result.index:
        squaredError.append(pow((result["targets"][i] - result["results"][i]),2))

    netAnalysisReport[-1][2] = sum(squaredError)/len(squaredError)

    result = pd.concat([result, pd.DataFrame(squaredError).rename(columns={0 : "squaredErr"})], axis=1)

    # Stampo il risultato in csv
    result.to_csv("results/" + str(neuralNet) + "_testOnTrainingSet.csv", index=False)

pd.DataFrame(netAnalysisReport, columns=["neuronsNumber","testSetMeanSquaredError","trainingSetMeanSquaredError"]).to_csv("results/netsAnalysisReport.csv", index=False)

# Costruisco un grafico scatter con gli errori quadratici medi a confronto
x = []
y1 = []
y2 = []

for i in netAnalysisReport:
    x.append(i[0])
    y1.append(i[1])
    y2.append(i[2])

pl.scatter(x,y1).set_color(c=(0,0,1)) #blue scatter for testSetMeanSquaredError
pl.scatter(x, y2).set_color(c=(1,0,0)) #red scatter for trainingSetMeanSquaredError
pl.xlabel("Neurons number")
pl.ylabel("Mean squared error")
pl.grid()
pl.show()



