import numpy as np
import pandas as pd
import neurolab as nl
import time

# Estraggo i dati dal file .csv e li inserisco in un dataset pandas.
dataset = pd.read_csv("ele-2.csv", header=None, names=["attr1", "attr2", "attr3", "attr4", "result"])

# Elimino eventuali tuple contenenti MISSING VALUE
dataset.dropna()

# Normalizzazione dei dati
datasetNorm = dataset.copy()
features = ["attr1", "attr2", "attr3", "attr4", "result"]
toNorm = dataset[features]
datasetNorm[features] = (toNorm - toNorm.min())/(toNorm.max() - toNorm.min())

# Estraggo e salvo su file informazioni importanti sul dataset
datasetInfo = pd.DataFrame(columns=["min","max"], 
                           data=[[dataset["attr1"].min(), dataset["attr1"].max()],
                                [dataset["attr2"].min(), dataset["attr2"].max()],
                                [dataset["attr3"].min(), dataset["attr3"].max()],
                                [dataset["attr4"].min(), dataset["attr4"].max()],
                                [dataset["result"].min(), dataset["result"].max()]])

datasetInfo.to_csv("results/datasetInfo.csv")

# Elimino eventuali duplicati
datasetNorm.drop_duplicates()

# Splitto il dataset in due dataset, uno per il training (contente il 70% delle tuple pescate casualmente) e uno per il test (contenente il 30% delle tuple restanti)
trainingSet = datasetNorm.sample(frac=0.7, random_state = 200)
testSet = datasetNorm.drop(trainingSet.index)

#Stampo a video le principali informazioni su training set e test set
print("dataset size: " + str(len(dataset)) + " tuples")
print("trainingSet absolute size: " + str(len(trainingSet)) + " tuples")
print("trainingSet percentage size: " + str(np.round((len(trainingSet) / len(dataset))*100, 0)) + " %")
print("testSet absolute size: " + str(len(testSet)) + " tuples")
print("testSet percentage size: " + str(np.round((len(testSet) / len(dataset))*100, 0)) + "%")

# Stampo in csv il testSet per verificare i risultati del training
testSet.to_csv("results/testSet.csv", index=False)

# Stampo in csv il trainingSet per una seconda verifica del training
trainingSet.to_csv("results/trainingSet.csv", index=False)

# Splitto il training in inputs e targets
datasetInputs = trainingSet[["attr1", "attr2", "attr3", "attr4"]]
datasetTargets = trainingSet[["result"]]

# Inizializzo la variabile contatore del numero di neuroni utilizzati dalla rete feed-forward
neuronsStartNumber = 5
neuronsMaxNumber = 50
neuronsIncrement = 5

# Inizializzo una lista di timer per appuntare il tempo di addestramento delle reti al variare del numero di neuroni
timeList = []
neuronsNumberList = []

# Fintanto che la variabile neuronsNumber è minore di 50
while (neuronsStartNumber <= neuronsMaxNumber):

    tStart = time.time()

    # Inizializzo la rete assegnandogli i parametri necessari
    # [[0,1], [0,1], [0,1], [0,1]]  ->  Numero di inputs e relativo insieme di appartenenza
    # [neuronsNumber, 1]            ->  numero di neuroni e numero di outputs
    # Successivamente cambio la funzione di trasferimento da quella di default (TanSig) a quella sigmoidale (LogSig)
    net = nl.net.newff([[0,1], [0,1], [0,1], [0,1]], [neuronsStartNumber, 1])
    net.layers[0].transf = nl.trans.LogSig()
    net.layers[1].transf = nl.trans.LogSig()

    # Avvio l'addestramento con il trainingSet assegnandogli i parametri necessari
    # datasetImputs         ->  n tuple composte da 4 attributi tutti appartenenti all'insieme [0,1]
    # datasetTargets        ->  n tuple composte da 1 attributo che rappresenta l'obbiettivo del training
    # epochs=1000           ->  numero di epoche di addestramento massimo
    # show=100              ->  ogni quante epoche visualizzare l'errore attuale a video
    # goal=0.01             ->  criterio di arresto basato sul valore dell'errore
    err = net.train(datasetInputs, datasetTargets, epochs=50, show=100, goal=0.01)

    tStop = time.time()

    #Aggiungo alla lista del numero di neuroni le informazioni relative alla n-esima iterazione
    neuronsNumberList.append(neuronsStartNumber)

    # Aggiungo alla lista la misurazione del tempo necessario ad addestrare la rete neurale corrente
    timeList.append(np.round(tStop - tStart,2))
    print(str(neuronsStartNumber) + " neurons training duration: " + str(np.round(tStop - tStart,2)) + " seconds")

    # Salvo la rete neurale su file
    net.save("results/" + str(neuronsStartNumber) + "NeuronsNet.net")

    #incremento il numero di neuroni della rete
    neuronsStartNumber += neuronsIncrement

# Stampo su file il totale dei tempi necessari all'addestramento delle varie reti
pd.concat([pd.DataFrame(timeList).rename(columns={0 : "time"}), pd.DataFrame(neuronsNumberList).rename(columns={0 : "neurons"})], axis=1).to_csv("results/timer.csv", index=False)


