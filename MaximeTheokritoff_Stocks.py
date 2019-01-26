"""
    Assignment 2 Python for Data Science
    Maxime Theokritoff
    """

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as anim
import matplotlib.dates as mdates
from collections import Counter
from mpl_toolkits import mplot3d




"""
    The following code is used to import and structure the stock data for analysis
    """
np.set_printoptions(threshold=np.nan)
"""
    This is to display the entier numpy arrays for debugging purposes
    """

print("Hello, please input a csv file with columns names: Date, Open, Close and Volume")
inputFile = input("Please entre your file path: ")
df = pd.read_csv(inputFile, sep=',',header=0, usecols=['Date', 'Open', 'Close', 'Volume'])
stockData = df.values
"""
    Importing the data as a pandas dataframe
    Converting the dataframe to an array
    """

"""
    The following functions give us usefull data needed for all other computations:
    findDayPos: returns the position of a date in an array.
    (The data imported has information on years irrelevent to our study.)
    purchaseShares: will buy for $1000 worth of shares on  '2016-12-30' close
    doNothing: Get the result of a 1 year buy and hold strategy.
    """

def findDayPos(myDate):
    for i in range(np.size(stockData,0)):
        if stockData[i,0] == myDate:
            position = i
    return position

def purchaseShares():
    position = findDayPos('2016-12-30')
    money = 1000
    shares =  int(money/stockData[position,2])
    money -=  stockData[position,2]* shares
    return [money,shares]

def doNothing():
    buy = findDayPos('2016-12-30')
    sell = findDayPos('2017-12-29')
    money = 1000
    shares =  int(money/stockData[buy,2])
    money +=  stockData[sell,2]* shares
    return money - 1000

"""---------------------------------------------------------------------------"""


"""
    1) Simple moving average crossover - The decision to buy or sell is based
    on the following computation based on the original daily stock information.
    Compute the average of the 50 previous closing prices. Compute the average of
    the 200 previous closing prices. If on the previous trading day the 50 price
    average was less than the 200 price average, and on this day the 200 price
    average is less than the 50 price average, this crossover indicates you
    should buy on the next day.  If on the previous trading day the 50 price
    average was more than the 200 price average, and on this day the 200
    price average is more than the 50 price average, this crossover
    indicates you should sell on the next day. If the relationship of the
    two averages remains the same between the days, then you should hold
    onto the stock with no changes the next day.
    """


"""
    SimpleMovingAvgCross takes 3 arguments:
    -stockData: Information on the stock studied
    -startDate: The first day of the trading period we are interested in
    -endDate: The last day of the trading period we are interested in
    and returns an array containing: "Date", "Open", "Average 50", "Average 200"
    """

def SimpleMovingAvgCross( startDate, endDate):
    start = findDayPos(startDate)-2
    end = findDayPos(endDate)
    AvgCross = np.array([["Date", "Open", "Average 50", "Average 200"]])
    for i in range(start, end+1):
        AvgCross = np.vstack((AvgCross, [stockData[i,0] , stockData[i,1] ,
                                         np.mean(stockData[i-50:i,2]) , np.mean(stockData[i-200:i,2])]))
    return AvgCross

smac = SimpleMovingAvgCross( '2017-01-03', '2017-12-29')
#print(smac)

"""
    smac = Simple Moving Average Crossover results
    """

"""
    2) Moving average crossover/divergence -  The decision to buy or sell is based
    on the following computation based on the original daily stock information.
    Moving averages will be used again, but in a slightly different way.
    First, you will be computing 12 and 26 day averages rather than 50 and 200.
    Also, rather than computing the simple average over these days, you will give
    a little more weight to the most recent closing price. This is called the
    exponential moving average (EMA). The 12 day EMA is computed as
    2 * Closingday12 / 13 + 11 * EMAday11 / 13. So the 12th day gets a double weighting
    in the calculation. The first day's EMA in a sequence can be computed as a simple
    average of the previous 12 days' closing prices to get things started.
    Similarly the 26 day EMA is computed as 2 * Closingday26 / 27 + 25 * EMAday25 / 27.
    You then compute another value called the MACD as the 12 day EMA - 26 day EMA
    for each day. From this difference, you can compute yet another value called
    the Signal as follows: SignaldayN = 2 * MACDdayN / 10 + 8 * SignaldayN-1 / 10.
    The first day's Signal value can be calculated as the simple average of the
    previous 9 days' MACD values to get things started. In order to make a decision,
    if on the previous trading day the MACD was less than the Signal value,
    and on this day the Signal value is less than the MACD, this crossover indicates
    you should buy on the next day.  If on the previous trading day the MACD was more
    than the Signal value, and on this day the Signal value is more than the MACD,
    this crossover indicates you should sell the next day. If the relationship of
    the two values remains the same between the days, then you should hold onto the
    stock with no changes the next day.
    """

def MovAvgCrossDiv( startDate, endDate):
    #Follows the formulas for Moving average with weighting
    start = findDayPos(startDate)-2
    end = findDayPos(endDate)
    AvgCross = np.array([["Date", "Open", "MACD", "Signal"]])
    for i in range(start, end+1):
        macd = (np.sum(stockData[i-12:i-1,2])+stockData[i-1,2]*2)/13-(np.sum(stockData[i-26:i-1,2])+ stockData[i-1,2]*2)/27
        macdMinus1 = (np.sum(stockData[i-11:i,2])+stockData[i,2]*2)/13-(np.sum(stockData[i-25:i,2])+stockData[i,2]*2)/27
        sig = 8/10*macd + 2/10*macdMinus1
        AvgCross = np.vstack((AvgCross, [stockData[i,0] , stockData[i,1] , macd , sig ]))
    #Stacks the data to an array
    return AvgCross

macd = MovAvgCrossDiv( '2017-01-03', '2017-12-29')
#print(macd)

"""
    buyOrSell will return data based on the followed startegy, SMAC or MACD.
    "Date", "Buy or Sell", "Shares", "Money"
    the last line of the array gives information on the end of year sell and has
    information on the overall profit of the strategy
    """

idx = []
"""
    Used to store index of Buy or Sales for SMAC
    """
idx2 =[]
"""
    Used to store index of Buy or Sales for MACD
    """

def buyOrSell(givenArray, endDate, idx):
    money = purchaseShares()[0]
    shares = purchaseShares()[1]
    """gets the current money available and share price"""
    spend = 0
    earned = 0
    """tracks our earnings and spend on 1 deal, money tracks for the year"""
    end = int(np.size(givenArray)/4)
    transactions = np.array([["Date", "Open", "Buy or Sell",  "Shares", "Money"]])
    for i in range(3,end):
        if givenArray[i-2,2]<givenArray[i-2,3] and givenArray[i-1,2]>givenArray[i-1,3] and money >= float(givenArray[i,1]):
            """IF a BUY condition is respected"""
            shares =  int(money/float(givenArray[i,1]))
            """sets the shares we can buy"""
            spent = float(givenArray[i,1])* shares
            """Calculates how much it will cost"""
            transactions = np.vstack((transactions,[givenArray[i,0],givenArray[i,1],"Bought", shares , spent ]))
            """Stacks the info about the deal"""
            money -=  float(givenArray[i,1])* shares
            """Decreases our money based on number of shares bought and their price"""
            idx.append(i)
        elif givenArray[i-2,2]>givenArray[i-2,3] and givenArray[i-1,2] < givenArray[i-1,3] and shares >= 1:
            """IF the sales condition is respected"""
            money +=  float(givenArray[i,1])* shares
            """Increases our money based on the share price"""
            earned = float(givenArray[i,1])* shares
            """Calculates total amount of sell"""
            transactions = np.vstack((transactions,[givenArray[i,0],givenArray[i,1],"Sold", shares, earned ]))
            """Stacks the deal info"""
            shares = 0
            """sets Shares to 0 as they are all sold"""
            idx.append(i)
    if (shares > 1):
        transactions = np.vstack((transactions,[givenArray[end-1,0],givenArray[end-1,1],"End of year sell",  0, money + shares * float(givenArray[int(end-1),1]) ]))
    """At the end of the year checks what the sale of all remain shares would yield."""
    return transactions


BOSsmac =buyOrSell(smac, '2017-12-29', idx)
BOSmacd = buyOrSell(macd, '2017-12-29', idx2)
#print(BOSsmac)
#print(BOSmacd)




"""
    1) Plot a graph of the closing price of the stock for the year you are analyzing.
    In the same figure, plot one curve depicting the 50 day average and another for the
    200 day average. Annotate on the plot those days where buy and sell actions should
    occur based on the crossover of the averages.
    """

def plottingSMAC(startDate, endDate):
    dates = np.array(smac[3:,0], dtype='datetime64[D]')
    avg50 = np.array(smac[3:,2], dtype=float)
    avg200 = np.array(smac[3:,3], dtype=float)
    start = findDayPos(startDate)-2
    end = findDayPos(endDate)
    
    close = np.array(stockData[start+2:end+1,2], dtype=float )
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    
    ax.plot(dates, close, label='Closing price', color ='#FECA30')
    ax.plot(dates, avg50, label='average 50 days', color ='#011E3A')
    ax.plot(dates, avg200, label='average 200 days', color ='#1B5190')
    
    count = 0
    for i in idx:
        if (count%2 == 0):
            ax.plot(dates[i], avg50[i], 'x', color = 'r')
            ax.annotate('Sell', xy=(dates[i], avg50[i]-0.15), color ='r',rotation=90,va='bottom')
            count += 1
        else:
            ax.plot(dates[i], avg50[i], 'x', color = 'g')
            ax.annotate('Buy', xy=(dates[i], avg50[i]+0.05), color ='g',rotation=90,va='bottom')
            count += 1

#ax.plot(BOSsmac[0,1:], BOSsmac[2,1:], 'x', color = '#FECA30')
plt.title('Simple Moving Average Crossover')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

plottingSMAC( '2017-01-03', '2017-12-29')

"""
    2) Plot a graph of the closing price of the stock for the year, and on the
    same figure plot curves for the 12 and 26 day EMA values. On a separate subplot,
    make graphs of the MACD value and the Signal value. Annotate on this plot those
    days where buy and sell actions should occur based on the crossover of these values.
    """

def SimpleMovingAvg1226( startDate, endDate):
    #identifies moving average cross from the 3 column of an array within a given periode.
    #We use satrat - 2 as we need to get the last trades of 2016 for the other calculations
    start = findDayPos(startDate)-2
    end = findDayPos(endDate)
    AvgCross = np.array([["Date", "Open", "Average 12", "Average 26", "Closing"]])
    for i in range(start, end+1):
        AvgCross = np.vstack((AvgCross, [stockData[i,0] , stockData[i,1] , np.mean(stockData[i-12:i,2]) ,
                                         np.mean(stockData[i-26:i,2]), stockData[i,2]]))
    return AvgCross
#Adds a row to our array following the Cross average formulas for every trading day of 2017

EMA = SimpleMovingAvg1226( '2017-01-03', '2017-12-29')




def plottingMACD():
    dates = np.array(EMA[1:,0], dtype='datetime64[D]')
    avg12 = np.array(EMA[1:,2], dtype=float)
    avg26 = np.array(EMA[1:,3], dtype=float)
    close = np.array(EMA[1:,4], dtype=float)
    
    MACD = np.array(macd[1:,2], dtype=float)
    signal = np.array(macd[1:,3], dtype=float)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    ax1.plot(dates, avg12, label='average 12 days', color ='#011E3A')
    ax1.plot(dates, avg26, label='average 26 days', color ='#1B5190')
    ax1.plot(dates, close, label='Closing price', color ='#FECA30')
    
    ax1.set_title('Moving Average Crossover/Divergence')
    ax1.set_ylabel('Stock Price')
    ax1.legend()
    ax1.tick_params(axis='x', which='both', bottom='off', top='off',labelbottom='off')
    
    ax2.plot(dates, MACD, label='MACD', color ='#011E3A')
    ax2.plot(dates, signal, label='Signal', color ='#FECA30', alpha=0.9)
    
    idx = np.argwhere(np.diff(np.sign(MACD - signal)) != 0)
    #ax.plot(dates[idx], avg50[idx], 'x', color = 'r')
    
    count = 0
    
    for n in idx:
        if (count%2 == 0):
            ax2.plot(dates[n], MACD[n], 'x', color = 'g')
            count += 1
        else:
            ax2.plot(dates[n], MACD[n], 'x', color = 'r')
            count += 1


plt.xlabel('Date')
plt.ylabel('Stock Price')
#sell = mpatches.Patch(color='r', label='Sell')
#buy = mpatches.Patch(color='m', label='Buy')
plt.legend()
    plt.show()

plottingMACD()

"""
    3. Relative Strength Index - The decision to buy or sell is based on the following
    computation based on the original daily stock information. The RSI for a given day,
    is computed as: RSIN = 100 - 100 / (1 + RSN). The value RSN is defined as
    Average gain / Average loss for the previous 14 trading day period. A gain or
    loss is determined by the difference in opening and closing prices on a day.
    If the closing price is higher it is a gain, otherwise it is a loss. For those
    days in the previous 14 day period on which there was a gain, take the average of
    these values. For those on which there was a loss, that that average, treating the
    losses as positive numbers in the calculation. If there are no losses during the
    14 day period, you can consider the value of RS to be infinite, meaning the RSI value
    will be 100. The value of the RSI indicator varies between 0 and 100. If the value falls
    below 30 on a day, this indicates that you should buy on the next day. If the value rises
    above 70 on a day, then you should sell the next day. On days between transitions above 70
    or below 30 you should hold onto the stock with no changes the next day.
    """

def RelStrInd( startDate, endDate):
    #Relative Strenght Index formulas
    start = findDayPos(startDate)-1
    end = findDayPos(endDate)
    #find the date we need to get data for
    AvgCross = np.array([["Date", "Open", "RS", "RSI"]])
    for i in range(start, end+1):
        count =0
        gain = 0
        loss = 0
        while count <14:
            #Check we are not over 14 values
            temp = stockData[i+count-14,2]-stockData[i+count-14,1]
            if temp >0:
                gain += temp
            elif temp < 0:
                loss += -temp
            count +=1
        if loss == 0:
            AvgCross = np.vstack((AvgCross, [stockData[i,0] , stockData[i,1] , avgGainLoss , 100]))
        else:
            avgGainLoss =(gain/14) / (loss/14)
            AvgCross = np.vstack((AvgCross, [stockData[i,0] , stockData[i,1] , avgGainLoss , 100-100/(1+avgGainLoss)]))
    return AvgCross

rsi = RelStrInd( '2017-01-03', '2017-12-29')

#print(rsi)

idxRSI =[]
"""
    Stores the index of where a buy or sell should occure
    """

def buyOrSellRsi(endDate):
    #Follows the exercice conditions for a buy or sell for RSI
    money = purchaseShares()[0]
    shares = purchaseShares()[1]
    spend = 0
    earned = 0
    end = int(np.size(rsi)/4)
    transactions = np.array([["Date", "Open", "Buy or Sell", "Shares", "Money"]])
    for i in range(2,end):
        if float(rsi[i-1,3])<30 and money >= float(rsi[i,1]):
            shares =  int(money/float(rsi[i,1]))
            spent = float(rsi[i,1])* shares
            transactions = np.vstack((transactions,[rsi[i,0],rsi[i,1],"Bought", shares , spent ]))
            money -=  float(rsi[i,1])* shares
            idxRSI.append(i)
        elif float(rsi[i-1,3])>70 and shares >= 1:
            money +=  float(rsi[i,1])* shares
            earned = float(rsi[i,1])* shares
            transactions = np.vstack((transactions,[rsi[i,0],rsi[i,1],"Sold", shares, earned ]))
            shares = 0
            idxRSI.append(i)
    if (shares > 1):
        transactions = np.vstack((transactions,[rsi[end-1,0],rsi[end-1,1],"End of year sell",  0, money + shares * float(rsi[int(end-1),1]) ]))
    return transactions

BOSrsi =buyOrSellRsi('2017-12-29')

#print(BOSrsi)

def gainLoss( startDate, endDate):
    #Relative Strenght Index formulas
    start = findDayPos(startDate)-1
    end = findDayPos(endDate)
    #find the date we need to get data for
    gainLoss = np.array([["Date","Gain Loss"]])
    for i in range(start, end-1):
        gainLoss = np.vstack((gainLoss,[stockData[i+1,0],stockData[i+1,2]-stockData[i,2]]))
    return gainLoss

#transactions = np.vstack((transactions,[givenArray[i,0],"Sold", shares, earned ]))

GorL = gainLoss( '2017-01-03', '2017-12-29')

#print(GorL)

"""
    3) Plot a bar graph of the daily gain/loss of the closing price throughout the year.
    Plot positive values for gains in one color and negative values for losses in another color.
    On a separate subplot graph the RSI daily value, along with horizontal lines representing
    RSI values of 30 and 70. Annotate on this plot those days where buy and sell actions
    should occur based on the RSI value.
    """

def RSIplotting():
    #Issues: import all columns individualy
    dates = np.array(rsi[1:,0], dtype='datetime64[D]')
    datesOfBOrS =  np.array(BOSrsi[1:,0], dtype='datetime64[D]')
    size = int(np.size(GorL)/2)
    bar_width = 0.1
    index =np.arange(size)
    
    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter('%b')
    
    
    
    height = 0
    ax = plt.subplot(211)
    for i in range(1,size):
        if float(GorL[i,1])<0.0:
            ax.bar(height,abs(float(GorL[i,1])),bar_width, label = 'Loss', color ='#011E3A' )
        else :
            ax.bar(height,float(GorL[i,1]),bar_width, label = 'Gain',color ='#FECA30')
        height = height + 0.05
    
    blue_patch = mpatches.Patch(label = 'Loss', color ='#011E3A')
    yellow_patch = mpatches.Patch(label = 'Gain',color ='#FECA30')


ax.set_ylabel('$', rotation=0)
ax.tick_params(axis='x', which='both', bottom='off', top='off',labelbottom='off')
plt.title('Daily gains or losses for 2017')
plt.legend(handles=[blue_patch, yellow_patch])

ax2 = plt.subplot(212)
ax2.plot(dates[1:],np.array([70 for i in range(size)]),color ='#FECA30')
ax2.plot(dates[1:],np.array([30 for i in range(size)]),color ='#FECA30')
ax2.plot(dates[1:], rsi[2:,3], label='RSI', color ='#011E3A')
ax2.xaxis.set_major_locator(months)
ax2.xaxis.set_major_formatter(monthsFmt)
ax2.set_ylabel('$', rotation=0)


count = 0
    for n in idxRSI:
        
        if (count%2 == 0):
            ax2.plot(dates[n-1], rsi[n-1,3], 'x', color = 'g')
            ax2.annotate('Sell', xy=(dates[n-1], rsi[n-1,3]), color ='g',rotation=90,va='bottom')
            count = count+ 1
        else:
            ax2.plot(dates[n-1], rsi[n-1,3], 'x', color = 'r')
            ax2.annotate('Buy', xy=(dates[n-1], rsi[n-1,3]), color ='r',rotation=90,va='bottom')
            count = count +  1



plt.title('RSI Values 2017')
plt.legend(loc=4)
plt.show()

RSIplotting()

"""
    4. On Balance Volume - The decision to buy or sell is based on the following computation
    based on the original daily stock information. The value of OBV for a given day is computed
    as follows, after being initialized at 0. If the closing price today is higher than
    the previous day's closing price, OBV = OBV + daily trading volume.
    If the closing price today is lower than the previous day's closing price,
    OBV = OBV - daily trading volume. If the closing price today is the same
    as the previous day, OBV does not change.
    
    
    Rather than looking for particular values of OBV, we want to consider how it is changing,
    that is we want to look at the general trend of the curve and look for places of maxima
    and minima. To do this we will fit a line to portions of the data and examine the line's
    slope. For a given day, we will fit a line to the value of OBV on the previous 20 days.
    This can be done using the polyfit function in numpy. If X is an array counting the days
    (1, 2, 3, etc.) and Y is the corresponding set of OBV values, then the function call
    np.polyfit(X,Y,1) will return a list with two values, the first of which is the slope
    of the line fit to the data. To recognize a maximum in the OBV plot, the slope
    will transition from a positive value to a negative value in consecutive days.
    This is an indicator that you should sell the next day. If the slope transitions
    from a negative value to a positive value in consecutive days, you have found a
    minimum in the plot, which is an indicator that you should buy the next day.
    Otherwise you should hold onto the stock with no changes the next day.
    
    """


def OnBalVol( startDate, endDate):
    #Follows the On Balance Volume Formula
    start = findDayPos(startDate)-22
    end = findDayPos(endDate)
    AvgCross = np.array([["Date", "Open",  "OBV", "Slope"]])
    x = np.arange(1,21)
    count = 0
    for i in range(start, end+1):
        obv = 0
        if stockData[i,2]>stockData[i-1,2]:
            obv += stockData[i,3]
        elif stockData[i,2]<stockData[i-1,2]:
            obv -= stockData[i,3]
        AvgCross = np.vstack((AvgCross, [stockData[i,0] , stockData[i,1] , obv, 0]))
    for i in range(21,int(np.size(AvgCross)/4)):
        AvgCross[i,3] = np.polyfit(x,AvgCross[i-20:i ,2].astype('float64'),1)[0]
    return AvgCross


OBV = OnBalVol( '2017-01-03', '2017-12-29')
#Follows the exercice guidlines to set up information for Buy/Sell conditions

#print(OBV)


def buyOrSellObv(givenArray, endDate):
    ##Follows the exercice conditions for a buy or sell for OBV
    money = purchaseShares()[0]
    shares = purchaseShares()[1]
    spend = 0
    earned = 0
    end = int(np.size(givenArray)/4)
    transactions = np.array([["Date", "Open", "Buy or Sell", "Shares", "Money"]])
    for i in range(20,end):
        if float(givenArray[i-2,3])<0 and float(givenArray[i-1,3])>0 and money >= float(givenArray[i,1]):
            shares =  int(money/float(givenArray[i,1]))
            spent = float(givenArray[i,1])* shares
            transactions = np.vstack((transactions,[givenArray[i,0],givenArray[i,1],"Bought", shares , spent ]))
            money -=  float(givenArray[i,1])* shares
        elif float(givenArray[i-2,3])>0 and float(givenArray[i-1,3])<0 and shares >= 1:
            money +=  float(givenArray[i,1])* shares
            earned = float(givenArray[i,1])* shares
            transactions = np.vstack((transactions,[givenArray[i,0],givenArray[i,1],"Sold", shares, earned ]))
            shares = 0
    if (shares > 1):
        transactions = np.vstack((transactions,[givenArray[end-1,0],givenArray[end-1,1],"End of year sell",  0, money + shares * float(givenArray[int(end-1),1]) ]))
    return transactions

BOSobv =buyOrSellObv(OBV, '2017-12-29')
#Applies the buy sell conditions

#print(BOSobv)

"""
    4) Plot a bar graph depicting the daily volume of your stock's trading. Overlay on this graph
    another bar graph showing the daily OBV value. Create an animation that shows how the slope of
    the 20 day OBV average varies throughout the year. Annotate on this plot those days where buy
    and sell actions should occur based on the changing value of this slope.
    """



def OBVplotting(startDate, endDate):
    start = findDayPos(startDate)
    end = findDayPos(endDate)
    graphData = np.array([["Date", "Vol","OBV"]])
    dates = np.array(stockData[1:,0], dtype='datetime64[D]')
    volume = np.array(stockData[1:,3], dtype=float)
    OBVavg = np.array( OBV[23:,3], dtype=float)
    dates2 = np.array( OBV[23:,0], dtype='datetime64[D]')
    
    blue_patch = mpatches.Patch(label = 'Volume traded', color ='#011E3A')
    yellow_patch = mpatches.Patch(label = 'OBV avg',color ='#FECA30')
    
    
    for i in range(start-1, end):
        graphData = np.vstack((graphData, [dates[i], volume[i],OBVavg[i-251]]))
    
    

    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter('%b')


ax1 = plt.subplot()
ax2 = ax1.twinx()

ax1.bar(graphData[1:,0], graphData[1:,1], color ='#011E3A')
ax1.xaxis.set_major_locator(months)
ax1.xaxis.set_major_formatter(monthsFmt)

ax2.bar(graphData[1:,0], graphData[1:,2], color ='#FECA30')


plt.legend(handles=[blue_patch, yellow_patch])
plt.title("Daily Volume and OBV Average")

plt.show()

OBVplotting('2017-01-03', '2017-12-29')



"""
    Annimation will not be done, didn't figure it outF
    
    
    def OBVanimation(num, data, line):
    line.set_data(data[..., :num])
    return line,
    
    fig1 =plt.figure()
    
    OBVValues = np.array( OBV[1:,2], dtype='float')
    OBVDates = np.array( OBV[1:,0], dtype='datetime64[D]')
    l, = plt.plot([], [], 'r-')
    
    plt.plot(OBVDates, OBVValues)
    
    line_ani = anim.FuncAnimation(fig1, OBVanimation, 25, fargs=(OBVValues, l), interval=50, blit=True)
    
    plt.show()
    
    # animation function.  This is called sequentially
    def animate(frame):
    if OBVavg[frame] - 5 < 0:
    xline = np.linspace(0, x[frame] + 5, 50)
    else:
    xline = np.linspace(x[frame] - 5, x[frame] + 5, 50)
    
    yline = deg15derval[frame] * xline + b[frame]
    lin.set_data(xline,yline)
    return lin,
    
    lin.set_data(dates, OBVavg)
    #ax7.plot(dates, OBVavg)
    return lin,
    
    #ani = anim.FuncAnimation(myFig, animate, frames=20, init_func=init_line, interval = 5000, repeat = False)
    #plt.show()
    
    """

"""
    Print out each buy and sell transaction indicating the day, the number of shares bought/sold, and the amount
    of money spent or earned. Also display the bank account values of the four methods to see which one did better
    at the end. As a last part of the comparison do the following. Sometimes any single indicator may not give the
    best results, but if multiple indicators say you should do something, chances are that suggestion is more accurate.
    Look at the suggested actions of each of the four methods. If two or more say buy, then buy. If two or more
    say sell, then sell. Otherwise, just hold the stock. Run your simulation for this combined strategy and print
    out the bank account value.
    """


def CompTwo(Recomendation1, Recomendation2):
    #This functions goal is to detect the days a buy/sell strategy is recomended
    #Compare specific values of two arrays to find matching data of the 1st columns and add 1 to the BUy or Sell column.
    #This counts the diiferent buys or sells
    #the function goes over every value of a control array with dates, open price and buy or sell as 0
    #It then loops through every row of the buy/sell recomendations
    #if it identifies identical dates, it checks if a BUY or a SELL is recomended and
    #adds 1 to the corrsponding date Buy or Sell on our control array
    MatchingStrategy = np.array([["Date", "Open", "Buy or Sell", "Shares", "Money"]])
    size = int(np.size(Recomendation1)/5)
    for i in range(1,size):
        if Recomendation1[i,0] in Recomendation2[:,0]:
            MatchingStrategy = np.vstack((MatchingStrategy, [Recomendation1[i,0], Recomendation1[i,1],Recomendation1[i,2],Recomendation1[i,3], Recomendation1[i,4]]))

return MatchingStrategy

def stackcompTwo():
    stack = np.array([["Date", "Open", "Buy or Sell", "Shares", "Money"]])
    stack = np.vstack((stack, CompTwo(BOSsmac,BOSmacd)[1:,:]))
    stack = np.vstack((stack, CompTwo(BOSsmac,BOSrsi)[1:,:]))
    stack = np.vstack((stack, CompTwo(BOSsmac,BOSobv)[1:,:]))
    
    stack = np.vstack((stack, CompTwo(BOSmacd,BOSrsi)[1:,:]))
    stack = np.vstack((stack, CompTwo(BOSmacd,BOSobv)[1:,:]))
    
    stack = np.vstack((stack, CompTwo(BOSrsi,BOSmacd)[1:,:]))
    return stack[stack[:,0].argsort()]

BOSmultiStartegy = stackcompTwo()

print(BOSmultiStartegy)



def buyOrSellFinal():
    #Implements the last Buy and sell strategy if more than 1 strategy makes a recomendation
    money = purchaseShares()[0]
    shares = purchaseShares()[1]
    end = int(np.size(BOSmultiStartegy)/5)
    datesEvent = np.array(BOSmultiStartegy[:end-1,0], dtype='datetime64[D]')
    action = np.array(BOSmultiStartegy[:end-1,0], dtype='datetime64[D]')
    spend = 0
    earned = 0
    transactions = np.array([["Date", "Open", "Buy or Sell", "Shares", "Money gained or payed"]])
    
    
    for i in range(end-1):
        if shares > 0 and BOSmultiStartegy[i,2] == "Sold":
            money +=  float(BOSmultiStartegy[i,1])* shares
            earned = float(BOSmultiStartegy[i,1])* shares
            transactions = np.vstack((transactions,[BOSmultiStartegy[i,0],BOSmultiStartegy[i,1],"Sold", shares, earned ]))
            shares = 0
        elif money >= float(BOSmultiStartegy[i,1])  and BOSmultiStartegy[i,2] == "Bought":
            shares =  int(money/float(BOSmultiStartegy[i,1]))
            spent = float(BOSmultiStartegy[i,1])* shares
            transactions = np.vstack((transactions,[BOSmultiStartegy[i,0],BOSmultiStartegy[i,1],"Bought", shares , spent ]))
            money -=  float(BOSmultiStartegy[i,1])* shares

if shares > 0:
    transactions = np.vstack((transactions,[BOSmultiStartegy[end-2,0],BOSmultiStartegy[end-2,1],"End of year sell",  0, money + shares * float(BOSmultiStartegy[int(end-2),1]) ]))
    
    return transactions

BOSFinal = buyOrSellFinal()

#print(BOSFinal)


"""
    5) Use your imagination to determine how best to present your comparison of the different techniques,
    especially showing how your $1 000 dollar investments fluctuate throughout the year.
    Think about using animation and maybe even some 3D graphing.
    On all of these graphs you should create proper titles,
    axis labels, legends, etc. to make them look nice.
    """



"""
    Outputs the data to a file
    """

"""
    def ComparingPlotting():
    
    fig = plt.figure(1)
    ax = plt.axes(projection = '3d')
    zcoord = ["BOS macd", "BOS smac", "BOS rsi", "BOS obv", "BOS final"]
    xcoord = []
    ycoord = []
    """

outF = np.array(["Advanced Python Assignment 1 Maxime Theokritoff", "","", "", ""])
outF = np.vstack((outF,["Simple moving average crossover", "","", "", ""]))
outF = np.vstack((outF,BOSsmac))
outF = np.vstack((outF,["Moving average crossover/divergence ","", "", "", ""]))
outF = np.vstack((outF,BOSmacd))
outF = np.vstack((outF,["Relative Strength Index ","", "", "", ""]))
outF = np.vstack((outF,BOSrsi))
outF = np.vstack((outF,["On Balance Volume ", "","", "", ""]))
outF = np.vstack((outF,BOSobv))

outF = np.vstack((outF,["Two or more indicators", "","", "", ""]))
outF = np.vstack((outF,BOSFinal))

"""Stacking the different strategy results into 1 array for export"""

df2 = pd.DataFrame(data=outF)
"""Converting our export array to a datafram"""

df2.to_csv('MaximeTheokritoff_Assignment2.csv', index = False, header = False)
"""Exporting data to CSV"""

