import pylab as pl
import neurolab as nl

input = [[0,0], [0,1], [1,0], [1,1]] #training set
target = [[0], [0], [0], [1]] #label del training set

net = nl.net.newp([[0,1],[0,1]], 1)

err = net.train(input, target, epochs=3, show=10, goal=0.1)

pl.figure()
pl.plot(err)
pl.xlabel("Epoch number")
pl.ylabel("training error")
pl.grid()
pl.show()