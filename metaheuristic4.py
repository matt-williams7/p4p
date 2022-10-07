# Imports
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
    #data["RSI2"] =  temp1.astype(int) - temp2.astype(int)
    data["RSI2"] = data["RSI"]



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
    sigScale = x[0]
    actionScale = x[1]
    linParams = x[2:]
    # portCont = x[]
    #sigCont = x[]

    # Get linear combination
    summ = sum(np.multiply(current,linParams))

    # Get sigmoid value
    sig = sigmoid(summ,sigScale)
    
    # Get portfolio force

    # Get final value
        # sigmoid*w1 + force*w2

    # Determine change in coins from action
    change = actionScale*sig
    
    # Extract portfolio info
    coins = portfolio[0]
    cash = portfolio[1]

    # If buying
    if change>0:
        # If cant afford
        if change*price+fee > cash:
            # Buy as much as you can
            return((coins + cash/(price*(1+fee)),0), price)
        # Otherwise return change
        else:
            #print(change)
            return((coins+change,cash-change*price*(1+fee)), price)
    # If selling
    else:
        # If selling more than we have
        if abs(change)>coins:
            # No change
            return((0,cash + coins*price*(1-fee)),price)
        # Otherwise return change
        else:
            return((coins+change, cash-change*price*(1-fee)),price)

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

    portfolio = initPort
    changes = np.zeros(nm)

    # Loop through months
    for month in train:
        portfolio = initPort

        # Loop through days
        for day in range(bch):

            # Get current and next day's data
            ind = month*bch+day
            current = data.iloc[ind]
            #next = data.iloc[ind+1]

            # Take action
            portfolio,finalPrice = action(x, current, portfolio, close[ind], fee)

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
    ret = np.sum(np.multiply(weights,changes))
    
    print(ret) 
    return(ret)

# Define function for getting a model's out-of-sample performance
def performSim(data, xopt, portfolio, nm, test, bch):
    
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
        # plt.title("Autotrader vs Buy&Hold Wealth")
        # plt.ylabel("Wealth ($)")
        # plt.xlabel("Date")
        # plt.legend()
        # plt.show()
        # plotVsBaseline(dates, wealths, ogWealths)
        #plotCoins(dates, coinss, close[nstart:nfinish]) 
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


# READ DATA =======================================
bch = 60
# Read in data
data_read = True
RSI = False
if data_read == True:
    data,close = readData("TI_data_btc1.csv", bch, RSI)

# MAKE TEST/TRAIN ======================

nd = data.shape[0]
nm = int(nd/bch)

ms = [i for i in range(nm)]

rand.seed(700)
msamp = rand.sample(ms,nm)
nmt = int(0.6*nm)
alltrain = msamp[0:nmt]
test = msamp[nmt:nm]

initPort = (1,close[0])

xopts = []
perfs = []

if (False):
# For each cval iteration
    for i in range(2):
        print("\n\n\n\n\n\n===========================\nITERATION ", i, "\n======================================\n\n\n\n")
        # Get train samp

        train = rand.sample(alltrain,nmt-int(0.2*nmt))

        # Training ------------

        # Make ub and lbs
        ub = [10 for i in range(data.shape[1]+2)]
        lb = [-x for x in ub]
        # sigscale
        lb[0]=0
        ub[0]=0.1 

        #actionscale
        lb[1]=0
        ub[1] = 1 
        
        

        # Fee
        fee = 0.01
        lam = 0.2
        # Run pso
        args = (data, initPort, close, fee, nm, train, lam,bch)
        xopt, fopt = pso(fitness, lb, ub, args=args, debug = True, maxiter=1)

        # Testing ------------------

        perf = performSim(data, xopt, initPort, nm, test,bch)

        xopts.append(xopt)
        perfs.append(perf)
        print("Performance is: ", perf)

#data_pd = {'xopts':xopts,'perfs':perfs}
#df = pd.DataFrame(data =data_pd)
#df.to_csv('D:\\OneDrive\\Documents\\Uni\\2022\\ENGSCI700\\MH\\xopts.csv')


print("yee")
#xopt = xopts[np.argmin(perfs)]

# Most recent best btc xopt:
xopt = [ 0.06100868, -7.71590182,  5.13015153,  4.58013229, -7.86504274, -0.51216633, -6.68232843,  6.78272049, -0.57935946]
print(xopt)


# TEST PLOTS =====================================================

# WHOLE DATA
nstart = 0
coinss, cashs, wealths, ogWealths, dates = testSim(data, xopt, initPort, nstart, nd,0.01)



plotVsBaseline(dates, wealths, ogWealths)
plotCoins(dates, coinss, close[nstart:nd]) 
plotParams(data,xopt)

#MONTHS
testSimMonth(data, xopt, initPort,bch,0.01, close)