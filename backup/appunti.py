#def linearRegression(X, Y):
    # X = dataset['attr1'].index.values.reshape(-1, 1)  # values converts it into a numpy array
    # Y = dataset.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    #linear_regressor = LinearRegression()  # create object for the class
    #linear_regressor.fit(X, Y)  # perform linear regression
    #Y_pred = linear_regressor.predict(X)  # make predictions
    #return Y_pred

#    /$$$$$$$   /$$$$$$  /$$$$$$$$ /$$$$$$         /$$$$$$  /$$       /$$$$$$$$  /$$$$$$  /$$   /$$ /$$$$$$ /$$   /$$  /$$$$$$ 
#   | $$__  $$ /$$__  $$|__  $$__//$$__  $$       /$$__  $$| $$      | $$_____/ /$$__  $$| $$$ | $$|_  $$_/| $$$ | $$ /$$__  $$
#   | $$  \ $$| $$  \ $$   | $$  | $$  \ $$      | $$  \__/| $$      | $$      | $$  \ $$| $$$$| $$  | $$  | $$$$| $$| $$  \__/
#   | $$  | $$| $$$$$$$$   | $$  | $$$$$$$$      | $$      | $$      | $$$$$   | $$$$$$$$| $$ $$ $$  | $$  | $$ $$ $$| $$ /$$$$
#   | $$  | $$| $$__  $$   | $$  | $$__  $$      | $$      | $$      | $$__/   | $$__  $$| $$  $$$$  | $$  | $$  $$$$| $$|_  $$
#   | $$  | $$| $$  | $$   | $$  | $$  | $$      | $$    $$| $$      | $$      | $$  | $$| $$\  $$$  | $$  | $$\  $$$| $$  \ $$
#   | $$$$$$$/| $$  | $$   | $$  | $$  | $$      |  $$$$$$/| $$$$$$$$| $$$$$$$$| $$  | $$| $$ \  $$ /$$$$$$| $$ \  $$|  $$$$$$/
#   |_______/ |__/  |__/   |__/  |__/  |__/       \______/ |________/|________/|__/  |__/|__/  \__/|______/|__/  \__/ \______/

#    /$$$$$$$   /$$$$$$  /$$$$$$$$ /$$$$$$        /$$$$$$ /$$   /$$ /$$$$$$$$ /$$$$$$$$  /$$$$$$  /$$$$$$$   /$$$$$$  /$$$$$$$$ /$$$$$$  /$$$$$$  /$$   /$$
#   | $$__  $$ /$$__  $$|__  $$__//$$__  $$      |_  $$_/| $$$ | $$|__  $$__/| $$_____/ /$$__  $$| $$__  $$ /$$__  $$|__  $$__/|_  $$_/ /$$__  $$| $$$ | $$
#   | $$  \ $$| $$  \ $$   | $$  | $$  \ $$        | $$  | $$$$| $$   | $$   | $$      | $$  \__/| $$  \ $$| $$  \ $$   | $$     | $$  | $$  \ $$| $$$$| $$
#   | $$  | $$| $$$$$$$$   | $$  | $$$$$$$$        | $$  | $$ $$ $$   | $$   | $$$$$   | $$ /$$$$| $$$$$$$/| $$$$$$$$   | $$     | $$  | $$  | $$| $$ $$ $$
#   | $$  | $$| $$__  $$   | $$  | $$__  $$        | $$  | $$  $$$$   | $$   | $$__/   | $$|_  $$| $$__  $$| $$__  $$   | $$     | $$  | $$  | $$| $$  $$$$
#   | $$  | $$| $$  | $$   | $$  | $$  | $$        | $$  | $$\  $$$   | $$   | $$      | $$  \ $$| $$  \ $$| $$  | $$   | $$     | $$  | $$  | $$| $$\  $$$
#   | $$$$$$$/| $$  | $$   | $$  | $$  | $$       /$$$$$$| $$ \  $$   | $$   | $$$$$$$$|  $$$$$$/| $$  | $$| $$  | $$   | $$    /$$$$$$|  $$$$$$/| $$ \  $$
#   |_______/ |__/  |__/   |__/  |__/  |__/      |______/|__/  \__/   |__/   |________/ \______/ |__/  |__/|__/  |__/   |__/   |______/ \______/ |__/  \__/

#    /$$$$$$$   /$$$$$$  /$$$$$$$$ /$$$$$$        /$$$$$$$  /$$$$$$$$ /$$$$$$$  /$$   /$$  /$$$$$$  /$$$$$$$$ /$$$$$$  /$$$$$$  /$$   /$$
#   | $$__  $$ /$$__  $$|__  $$__//$$__  $$      | $$__  $$| $$_____/| $$__  $$| $$  | $$ /$$__  $$|__  $$__/|_  $$_/ /$$__  $$| $$$ | $$
#   | $$  \ $$| $$  \ $$   | $$  | $$  \ $$      | $$  \ $$| $$      | $$  \ $$| $$  | $$| $$  \__/   | $$     | $$  | $$  \ $$| $$$$| $$
#   | $$  | $$| $$$$$$$$   | $$  | $$$$$$$$      | $$$$$$$/| $$$$$   | $$  | $$| $$  | $$| $$         | $$     | $$  | $$  | $$| $$ $$ $$
#   | $$  | $$| $$__  $$   | $$  | $$__  $$      | $$__  $$| $$__/   | $$  | $$| $$  | $$| $$         | $$     | $$  | $$  | $$| $$  $$$$
#   | $$  | $$| $$  | $$   | $$  | $$  | $$      | $$  \ $$| $$      | $$  | $$| $$  | $$| $$    $$   | $$     | $$  | $$  | $$| $$\  $$$
#   | $$$$$$$/| $$  | $$   | $$  | $$  | $$      | $$  | $$| $$$$$$$$| $$$$$$$/|  $$$$$$/|  $$$$$$/   | $$    /$$$$$$|  $$$$$$/| $$ \  $$
#   |_______/ |__/  |__/   |__/  |__/  |__/      |__/  |__/|________/|_______/  \______/  \______/    |__/   |______/ \______/ |__/  \__/

#    /$$$$$$$   /$$$$$$  /$$$$$$$$ /$$$$$$        /$$$$$$$$ /$$$$$$$   /$$$$$$   /$$$$$$  /$$$$$$$$ /$$$$$$  /$$$$$$$  /$$      /$$  /$$$$$$  /$$$$$$$$ /$$$$$$  /$$$$$$  /$$   /$$
#   | $$__  $$ /$$__  $$|__  $$__//$$__  $$      |__  $$__/| $$__  $$ /$$__  $$ /$$__  $$| $$_____//$$__  $$| $$__  $$| $$$    /$$$ /$$__  $$|__  $$__/|_  $$_/ /$$__  $$| $$$ | $$
#   | $$  \ $$| $$  \ $$   | $$  | $$  \ $$         | $$   | $$  \ $$| $$  \ $$| $$  \__/| $$     | $$  \ $$| $$  \ $$| $$$$  /$$$$| $$  \ $$   | $$     | $$  | $$  \ $$| $$$$| $$
#   | $$  | $$| $$$$$$$$   | $$  | $$$$$$$$         | $$   | $$$$$$$/| $$$$$$$$|  $$$$$$ | $$$$$  | $$  | $$| $$$$$$$/| $$ $$/$$ $$| $$$$$$$$   | $$     | $$  | $$  | $$| $$ $$ $$
#   | $$  | $$| $$__  $$   | $$  | $$__  $$         | $$   | $$__  $$| $$__  $$ \____  $$| $$__/  | $$  | $$| $$__  $$| $$  $$$| $$| $$__  $$   | $$     | $$  | $$  | $$| $$  $$$$
#   | $$  | $$| $$  | $$   | $$  | $$  | $$         | $$   | $$  \ $$| $$  | $$ /$$  \ $$| $$     | $$  | $$| $$  \ $$| $$\  $ | $$| $$  | $$   | $$     | $$  | $$  | $$| $$\  $$$
#   | $$$$$$$/| $$  | $$   | $$  | $$  | $$         | $$   | $$  | $$| $$  | $$|  $$$$$$/| $$     |  $$$$$$/| $$  | $$| $$ \/  | $$| $$  | $$   | $$    /$$$$$$|  $$$$$$/| $$ \  $$
#   |_______/ |__/  |__/   |__/  |__/  |__/         |__/   |__/  |__/|__/  |__/ \______/ |__/      \______/ |__/  |__/|__/     |__/|__/  |__/   |__/   |______/ \______/ |__/  \__/

#pl.figure()
    #pl.plot(err)
    #pl.xlabel("Epoch number")
    #pl.ylabel("training error")
    #pl.grid()
    #pl.show()