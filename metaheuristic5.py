# Imports
from mimetypes import init
from pickle import TRUE
import pandas as pd
from pyswarm import pso
import matplotlib.pyplot as plt
import numpy as np
import random as rand

# FUNCTIONS ====================================

# Define function for reading in data
def readData(file, bch, RSI =False):
    ''' 

    '''
    # Read in data
    data = pd.read_csv(file, index_col='date', parse_dates=True)
    data.sort_index(inplace=True)

    # Modify technical indicator columns
    if (RSI==False):
        temp1 = data["Aroon up"]==100
        temp2 =data["Aroon down"]==100
        data["Aroon2"] = temp1.astype(int)-temp2.astype(int)
        data["MACD2"] = data["MACD hist"]/abs(data["MACD hist"]) 
        data["ADX2"] = data["ADX"]>40
        # Modify , LR, PR, OBV
        data["OBV"] = (data["OBV"] - np.mean(data["OBV"]))/np.std(data["OBV"])
    else:
        data.drop(["Log Return", "Percentage Return", "OBV"], axis=1, inplace=True)

    temp1 = data["RSI"]<30
    temp2 = data["RSI"]>70
    data["RSI2"] =  temp1.astype(int) - temp2.astype(int)

    # Extract close vector
    close = data["close"]

    # Drop unused columns
    drop = ["date.1","open","high","low", "close", "volume", "DI_pos", "DI_neg", "Aroon oscilator", "Aroon down", "Aroon up", "MACD", "MACD hist", "MACD signal", "RSI","ADX"] 
    data.drop(drop, axis=1, inplace=True)

    # Drop rows to have nrows a multiple of batch size
    nrow = data.shape[0]
    data = data.iloc[nrow%bch:nrow]
    
    # Return new dataframe and close vector
    return (data, close)

# Define function to return shifted sigmoid 
def sigmoid(x,a):
    z = np.exp(-a*x)
    sig = 2 / (1 + z) -1
    return sig

# Define function for taking action for current timestep
def action(x, current, portfolio, price, fee):
    # Extract parameters
    sigScale = x[0]
    actionScale = x[1] # max proportion of portfolio to trade
    linParams = x[2:]

    # Extract portfolio info
    coins = portfolio[0]
    cash = portfolio[1]

    # Get linear combination
    summ = sum(np.multiply(current,linParams))
    # Get sigmoid value
    sig = sigmoid(summ,sigScale)
    # Determine change in coins from action
    portwealth = cash+coins*price
    change = actionScale*portwealth*sig/price
    
    # If buying
    if change>0:
        # If cant afford
        if change*price+fee > cash:
            # Buy as much as you can
            return((coins + cash/(price*(1+fee)),0), price,cash/(price*(1+fee)))
        # Otherwise return change
        else:
            #print(change)
            return((coins+change,cash-change*price*(1+fee)), price,change)
    # If selling
    else:
        # If selling more than we have
        if abs(change)>coins:
            # No change
            return((0,cash + coins*price*(1-fee)),price,-coins)
        # Otherwise return change
        else:
            return((coins+change, cash-change*price*(1-fee)),price,change)


# Define sim-opt fitness function
def fitness(x, *args):
    '''
    x = params
    args = (data, initial_portfolio = (coins,cash), close, fee, #month, train months, lambda, batch size)
    '''
    
    # Extract args
    data = args[0] # dataframe
    n = data.shape[0]
    initPort = args[1] # initial portfolio
    close = args[2] # price vector
    fee = args[3] # fee
    nm = args[4] # number of batches
    train = args[5] # train
    lam = args[6] #lambda
    bch = args[7] #batch size

     # Initialise output vector
    changes = np.zeros(nm)

    # Loop through months
    for month in train:
        # Reset initial portfolio
        portfolio = initPort

        # Loop through days
        for day in range(bch):

            # Get current and next day's data
            ind = month*bch+day
            current = data.iloc[ind]

            # Take action
            portfolio,finalPrice,act = action(x, current, portfolio, close[ind], fee)

        # Append month's performance to others    
        changes[month] = (-portfolio[0]+initPort[0])*finalPrice - portfolio[1] + initPort[1]

    # If using CVar
    if lam>0:
        # Set up weights
        arg_max = np.argmax(changes)
        weights = [(1-lam)/(nm-1) for i in range(nm) ]
        weights[arg_max] = lam

    # Otherwise
    else:
        # Equal weights
        weights = [1/nm for i in range(nm)]

    # Return weighted month performances
    fitness = np.sum(np.multiply(weights,changes))
    
    # Print and return fitness
    print(fitness) 
    return(fitness)

# Define function for getting a model's out-of-sample performance
def performSim(data, xopt, portfolio, nm, test, bch, fee):
    
    # Initialize array for monthly performance
    changes = np.zeros(nm)

    # For each month
    for month in test:

        # For each day
        for day in range(bch):
            # Get current day and its data
            ind = month*bch+day
            current = data.iloc[ind]

            # Take action
            portfolio,finalPrice = action(xopt, current, portfolio, close[ind], fee)

        # Append month's performance to others
        changes[month] = (-portfolio[0]+initPort[0])*finalPrice - portfolio[1] + initPort[1]

    # Return OOB performance as monthly performance average
    return(np.mean(changes))


# Define function for simulating trader on a subset of data
def testSim(data, xopt, portfolio, nstart, nfinish, fee):

    # Get number of days in simulatio
    n2 = nfinish-nstart

    # Set up output arrays
    coinss = np.zeros(n2)
    cashs = np.zeros(n2)
    wealths = np.zeros(n2)
    ogWealths = np.zeros(n2)
    
    # For each day
    for ind in range(nstart,nfinish):
        # Get current day
        current = data.iloc[ind]

        # Take action
        portfolio,finalPrice = action(xopt, current, portfolio, close[ind], fee)

        # Update current coins and cash
        coins = portfolio[0] 
        cash = portfolio[1]

        # Update history vector of coins, cash, wealth, and baseline wealths
        coinss[ind-nstart] = coins
        cashs[ind-nstart] = cash
        wealths[ind-nstart] = coins*finalPrice + cash
        ogWealths[ind-nstart] = close[0]+close[ind]

    # Get dates of subset to return
    dates = data.index[nstart:nfinish]

    return(coinss, cashs, wealths, ogWealths,dates)


# Function for simulating/plotting on months
def testSimMonth(data, xopt, portfolio,bch,fee, close):

    fig,axs = plt.subplots(2,4)
    months = [i for i in range(0,int(data.shape[0]/bch),3)]
    for i in range(8):
        month = months[i]
        nstart = month*bch
        nfinish = month*bch+bch
        coinss, cashs, wealths, ogWealths, dates = testSim(data, xopt, portfolio, nstart,nfinish, fee)
        axs[int(i/4),(i%4)].plot(dates, wealths, 'b', label = 'Wealth')
        axs[int(i/4),(i%4)].plot(dates, ogWealths, 'r', label = "Baseline")
 
    plt.show()
    fig1,axs1 = plt.subplots(2,4)
    for i in range(len(months)-1):
        month = months[i]
        nstart = month*bch
        nfinish = month*bch+bch
        coinss, cashs, wealths, ogWealths, dates = testSim(data, xopt, portfolio, nstart,nfinish, fee)
        axs1[int(i/4),(i%4)].plot(dates, coinss, 'y', label = 'Coins')
        axs1[int(i/4),(i%4)].plot(dates, close[nstart:nfinish]/np.mean(close[nstart:nfinish]), 'r', label = 'Price')
    plt.show()

# Define plotting functions
def plotVsBaseline(dates, wealths, ogWealths):
    plt.plot(dates, wealths, 'b', label = 'Wealth')
    plt.plot(dates, ogWealths, 'r', label = "Baseline")
    plt.title("Autotrader vs Buy&Hold Wealth")
    plt.ylabel("Wealth ($)")
    plt.xlabel("Date")
    plt.legend()
    plt.show()

def plotCoins(dates, coinss, close):
    plt.plot(dates, coinss, 'y', label = 'Coins')
    plt.plot(dates, close/np.mean(close), 'r', label = 'Price')
    plt.xlabel("Date")
    plt.title("Number of coins vs Normalised coin value")
    plt.legend()
    plt.show()

# Function for plotting parameters
def plotParams(data, xopt):

    names = ['Sig scale', 'ActionScale']
    names.extend(list(data.columns.values))
    plt.bar(names,xopt)
    for i in range(len(xopt)):
        val = xopt[i]
        plt.text(i-0.25,val/2,val)
    plt.xlabel("Parameters")
    plt.ylabel("Parameter values")
    plt.title("Parameter Values")
    plt.show()


# READ DATA ====================================================

# Read in data
data_read = True
RSI = False
bch = 30 # Batch size
if data_read == True:
    data,close = readData("TI_data_btc1.csv", bch, RSI)

# Use data after 2020
data = data.iloc[data.index.year>=2020]

# MAKE TEST/TRAIN ===============================================
# Get number of days and months in the data
nd = data.shape[0]
nm = int(nd/bch)

# Make test and train sets of months
ms = [i for i in range(nm)]
rand.seed(700)
msamp = rand.sample(ms,nm)
nmt = int(0.5*nm)
alltrain = msamp[0:nmt]
test = msamp[nmt:nm]


# TRAIN MODELS ===================================================
# 
initPort = (1,close[0])

xopts = []
perfs = []

# Number of models to train
nmodels = 50
# Trading fee
fee = 0.01
# CVar lamda weight on worst month
lam = 0.2

if (True):
# For each model iteration
    for i in range(nmodels):
        print("\n\n\n\n\n\n===========================\nModel ", i, "\n======================================\n\n\n\n")

        # Get train samp
        train = rand.sample(alltrain,nmt-int(0.2*nmt))

        # Make ub and lbs
        ub = [10 for i in range(data.shape[1]+2)]
        lb = [-x for x in ub]
        # sigscale
        lb[0]=0.01
        ub[0]=0.1 
        #actionscale
        lb[1]=0.01
        ub[1] = 1 
    
        # Run pso
        args = (data, initPort, close, fee, nm, train, lam, bch)
        xopt, fopt = pso(fitness, lb, ub, args=args, debug = True, maxiter=20)

        # Add model to list
        xopts.append(xopt)

# SIMULATION ============================================
# Get number of testing months
nmt = len(test)

# Initialise lists for monthly performances for baseline, each model, and average model
base = [] 
modperfs = []
average = []
average2 = []

# For each testing month
for i in range(nmt):
    # Get current month
    month = test[i]

    # Initialize current average portfolio for two averaging methods
    avgPort = (1, close[month*bch])
    avgPort2 = (1, close[month*bch])
    
    # List for current portfolio of each model
    modPorts = [(1, close[month*bch]) for i in range(nmodels)]
    # List of model's portfolios to average
    avgPorts = [(1, close[month*bch]) for i in range(nmodels)]
    # List of model's actions to average
    modActs = [0 for i in range(nmodels)]

    # Get baseline change in wealth

    # Simulate each model and average model through each month
    for day in range(bch):

        # Get current day and its data
        ind = month*bch+day
        current = data.iloc[ind]
        price = close[ind]

        # For each model
        for mod in range(nmodels):
            # Get current model
            x = xopts[mod]

            # Get model's new portfolio from action on current average portfolio
            newModPort, fp,act = action(x, current, modPorts[mod], price, fee)
            modPorts[mod] = newModPort

            # Get model's new portfolio from action on current model's portfolio
            newAvgPort, fp, act = action(x, current, avgPort, price, fee)
            avgPorts[mod] = newAvgPort
            modActs[mod] = act
        
        # Get average portfolio method 1
        coin = np.mean([i[0] for i in avgPorts])
        cash = np.mean([i[1] for i in avgPorts])
        avgPort = (coin,cash)

        # Get average portfolio method 2
        meanAct = np.mean(modActs)
        avgPort2 = (avgPort[0]+meanAct, avgPort[1]-meanAct*price*(1-fee))

    # Add performances ** comparing wealth at end to wealth at start:
    # (not hold wealth at end to trader wealth a end like above)
    # update initport to be star of month? or just a sum?

    # Starting wealth
    iWealth = 2*close[month*bch]

    # Baseline
    base.append(price - close[month*bch])
    lis = [por[0]*price + por[1] - iWealth for por in modPorts]
    modperfs.append(lis)
    average.append(avgPort[0]*price + avgPort[1]-iWealth)
    average2.append(avgPort2[0]*price + avgPort2[1]-iWealth)

# Write to csv =================================================================\

# Baseline and averages performance
df1 = pd.DataFrame([base,average,average2])
df1 = df1.transpose()
df1.columns = ["Baseline", "AveragePort", "AverageAct"]
df1.to_csv("base&avgPerfs.csv", index=False)

# Model performances
df2 = pd.DataFrame(modperfs)
df2.columns = ["Model"+str(i) for i in range(nmodels)]
df2.to_csv("modPerfs.csv", index=False)

# Models
df3 = pd.DataFrame(xopts)
df3 = df3.transpose()
df3.columns = ["Model"+str(i) for i in range(nmodels)]
df3.to_csv("modelParams.csv",index=False)