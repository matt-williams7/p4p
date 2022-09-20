# Imports
from pickle import TRUE
import pandas as pd
from pyswarm import pso
import matplotlib.pyplot as plt
import numpy as np

# FUNCTIONS ====================================

# Define function for reading in data
def readData(file, RSI =False):
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
        # Modify close, LR, PR, OBV
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

    # Return new dataframe and close vector
    return (data, close)

# Define function to return shifted sigmoid 
def sigmoid(x,a):
    z = np.exp(-a*x)
    sig = 2 / (1 + z) -1
    return sig

# Define function for taking action for current timestep
def action(x, current, portfolio, price, fee,fee_bool):
    sigScale = x[0]
    actionScale = x[1]
    linParams = x[2:]
    #sigScale = x[0]
    # portCont = x[]
    #sigCont = x[]
    #actionParam = x[2]

    # Get linear combination
    summ = sum(np.multiply(current,linParams))

    # Get sigmoid value
    sig = sigmoid(summ,sigScale)

    # Get portfolio force

    # Get final value
        # sigmoid*w1 + force*w2
    
    # Determine action from value
    

    # Determine change in coins from action
    change = actionScale*sig
    
    # Extract portfolio info
    coins = portfolio[0]
    cash = portfolio[1]

    # If buying
    if change>0:
        # If cant afford
        if fee_bool == True:
            if change*price+fee > cash:
                # Buy as much as you can
                return((coins + cash/(price*(1+fee)),0), price)
            # Otherwise return change
            else:
                #print(change)
                return((coins+change,cash-change*price*(1+fee)), price)
        else:
            if change*price > cash:
            # Buy as much as you can
                return((coins + cash/(price),0), price)
        # Otherwise return change
            else:
            #print(change)
                return((coins+change,cash-change*price), price)
    # If selling
    else:
        if fee_bool == True:
        # If selling more than we have
            if abs(change)>coins:
                # No change
                return((0,cash + coins*price*(1-fee)),price)
            # Otherwise return change
            else:
                return((coins+change, cash-change*price*(1-fee)),price)
        else:
            if abs(change)>coins:
                # No change
                return((0,cash + coins*price),price)
            # Otherwise return change
            else:
                return((coins+change, cash-change*price),price)

# Define sim-opt fitness function
def fitness(x, *args):
    '''
    x = params
    args = (data, initial_portfolio = (coins,cash), close, fee)
    '''
    
    # Extract args
    data = args[0]
    n = data.shape[0]
    initPort = args[1]
    close = args[2]
    fee = args[3]
    portfolio = initPort
    changes = np.zeros(int(n/30))
    # Loop through months
    for month in range(int(n/60)):
        portfolio = initPort
        # Loop through days
        for day in range(30):

            # Get current and next day's data
            ind = month*30+day
            current = data.iloc[ind]
            #next = data.iloc[ind+1]

            # Take action
            portfolio,finalPrice = action(x, current, portfolio, close[ind], fee, fee_bool)
            
        changes[month] = (-portfolio[0]+initPort[0])*finalPrice - portfolio[1] + initPort[1]
    print(np.mean(changes))
    return(np.mean(changes))


# READ DATA =======================================

# Read in data
data_read = True
RSI = True
if data_read == True:
    data,close = readData("TI_data.csv", RSI)

# FIT ===============================================

# Make ub and lbs
ub = [1000 for i in range(data.shape[1]+2)]
lb = [-x for x in ub]

# Fee
fee = 0.01
fee_bool = TRUE

# Run pso
args = (data,(1,close[0]), close, fee,fee_bool)
#xopt, fopt = pso(fitness, lb, ub, args=args, debug = True, maxiter=8)

#print(xopt)
#print(fopt)

# update to be sets of profits in 30 day periods, save for each outer for loop

#[close,volume,logreturn,percentagereturn,OBV, aroonbinary,aroonb2, macdbin, RSI1bin, RSI2bin, ADX2bin]
#xopt = [-1.00000000e-03,  7.83412484e-08, -7.83650181e-01,  4.75011697e-01, 8.94889699e-09,  8.88322994e-01,  8.17160345e-01, -7.69909878e-01, -1.00000000e+00, -2.61221500e-01, -1.92637346e-01]

#xopt = [-1.00000000e-03, -8.24627008e-07, -4.08679017e-01, -1.00000000e+00, 7.38470713e-08, -2.00156069e-01, -2.75849342e-01, -7.67933063e-01, 6.50557337e-01,  2.64987828e-01,  5.18420536e-01]

#xopt = [  710.8384057,   -950.06709355,   -21.51516912,  -963.26508504, -1000, 758.73503134,  -226.24275816]

# xopt = [-782.760197,   -18741.76458582, -11033.87422626,  24382.07439214, 9624.6234799,  -70646.61920364, -16002.13638915,  27560.46970967, -45224.23673756]

#xopt with fee of 0.1
#xopt = [ 100000, 46005.44563952, -13062.14386056, 19395.57527876, -31810.97743726, -100000, 35960.52434828, 73143.6729333, -86243.24895243]

#xopt = [ 14103.19537751, 16143.54669374, -4951.91813066, 52975.42537164, 9014.75928065, -57164.95668836, -62689.76194088, -23160.74809452, 40204.74441736]

xopt = [-736.66114168,  231.86343178, -307.5781159 ]

# TEST =====================================================

# Set up test simulation
portfolio = (1,close[0])



# Define function for testing
def testSim(data, xopt, portfolio):
    n = data.shape[0]
    coinss = np.zeros(n)
    cashs = np.zeros(n)
    wealths = np.zeros(n)
    ogWealths = np.zeros(n)
    

    for ind in range(n):
        #portfolio = (1, data.close[month])
    

        # Get current and next day's data
        #ind = month*30+day
        current = data.iloc[ind]
        #next = data.iloc[ind+1]

        # Take action
        portfolio,finalPrice = action(xopt, current, portfolio, close[ind], fee, fee_bool)

        # Update current coins and cash
        coins = portfolio[0] 
        cash = portfolio[1]
        # Update history vector of coins, cash, wealth, and baseline wealths
        coinss[ind] = coins
        cashs[ind] = cash
        wealths[ind] = coins*finalPrice + cash
        ogWealths[ind] = close[0]+close[ind]
    return(coinss, cashs, wealths, ogWealths)

coinss, cashs, wealths, ogWealths = testSim(data, xopt, portfolio)


# PLOT =================================

dates = data.index

def plotVsBaseline(dates, wealths, ogWealths):
    plt.plot(dates, wealths, 'b', label = 'Wealth')
    plt.plot(dates, ogWealths, 'r', label = "Baseline")
    plt.legend()
    plt.show()

def plotCoins(dates, coinss, close):
    plt.plot(dates, coinss, 'y', label = 'Coins')
    plt.plot(dates, close/np.mean(close), 'r', label = 'Price')
    plt.legend()
    plt.show()

plotVsBaseline(dates, wealths, ogWealths)
plotCoins(dates, coinss, close)


