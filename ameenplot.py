#software to generate plots from .csv files using matplotlib
#just a prototype for now 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys

from scipy.fft import set_backend




print(f"Name of the script      : {sys.argv[0]=}")
print(f"Arguments of the script : {sys.argv[1:]=}")

print(len(sys.argv))

plotmode =0
#plot modes 1 = stacked, plot mode 2 = combined

if len(sys.argv)==1:
    print('Missing arguments: file name, plot type')

elif len(sys.argv)==2:
    plotmode=1
elif len(sys.argv)==3:
    if sys.argv[2] == 'combined':
        plotmode = 2
    elif sys.argv[2] == 'separate':
        plotmode = 3
    else:
        plotmode = 1
elif len(sys.argv)==4:
    if sys.argv[2] == 'combined':
        plotmode = 2
    elif sys.argv[2] == 'separate':
        plotmode = 3
    else:
        plotmode = 1
    relevantData = sys.argv[3]
else:
    print('Too many arguments')


print('plot mode type is '+str(plotmode))

inputFile = str(sys.argv[1])
inputFileCopy=pd.read_csv(inputFile, sep=',',header=None)

numberColumns = (inputFileCopy).count(axis='columns')[1]
numberRows = (inputFileCopy).count(axis='rows')[1]
#print('number of columns ',numberColumns )
#print('number of rrolumns ',numberRows )

#print('number of columns ',type(inputFileCopy))
time=inputFileCopy[0][1:].astype(float)/1e-12 #scale to ps
titles = {}
for cols in range(0,numberColumns):
    #print(inputFileCopy[cols][0])
    title = inputFileCopy[cols][0]
    titles.update({(cols):title})

print(titles)

#create number of subplots
numberPlots = numberColumns-1
position = range(1,numberPlots + 1)

#fig=plt.figure(1)
#print('number of plots:', numberPlots)
if numberColumns==2:
    plotmode=3

if plotmode==1: #stacked

    fig,ax=plt.subplots(numberPlots)
    #print(inputFileCopy[0][0:])
#WAS 1,NUMBERPLOTS
    for k in range(1,numberPlots+1):
    
    #print(k)
        #ax[k].plot(time,inputFileCopy[k+1][1:].astype(float)) 
    #print(titles[k])
        #print(numberPlots,'SUBPLOTS')
        if titles[k][0] == 'I':
        #print('current')
           # yplot= inputFileCopy[k][1:].astype(float)*1000
            ax[k-1].plot(time,inputFileCopy[k][1:].astype(float)*1e6) 

            ax[k-1].set_ylabel('I [\N{GREEK SMALL LETTER MU}A]')
            #print(titles[k][0])
        elif titles[k][0] == 'P':
        #print('phase')
            ax[k-1].set_ylabel('\N{GREEK SMALL LETTER PHI} [rad]')
            #yplot= inputFileCopy[k][1:].astype(float)
            ax[k-1].plot(time,inputFileCopy[k][1:].astype(float)) 
            #print(titles[k][0])


        elif titles[k][0] == 'V':
        #print('voltage')
            #ax[k-1].set_ylabel('V [mV]')
            #yplot= inputFileCopy[k][1:].astype(float)
            #ax[k-1].plot(time,inputFileCopy[k][1:].astype(float)*1E6)
            #print(titles[k][0])

            if max(inputFileCopy[k][1:].astype(float))<0.1:
                ax[k-1].set_ylabel('V [mV]')
                ax[k-1].plot(time,inputFileCopy[k][1:].astype(float)*1E3) 
            else:
                ax[k-1].set_ylabel('V [V]')
                ax[k-1].plot(time,inputFileCopy[k][1:].astype(float)) 
        ax[k-1].set_title(titles[k])
        #else:
            #ax[k].remove()


    ax[numberPlots-1].set_xlabel('t [ps]')
    fig.tight_layout()

    #ax[0].remove()
    plt.show()

elif plotmode==2: #combined
    plottypes=['-','--','-.',':']
    plottitle=input('Input Title for Plot: ' )
    legend=[]
    fig=plt.figure()
    for k in range(1,numberPlots+1):
        if titles[k][0] == 'I':
            plt.plot(time,inputFileCopy[k][1:].astype(float)*1E6,plottypes[k%len(plottypes)-1]) 
            plt.ylabel('I [\N{GREEK SMALL LETTER MU}A]')

        elif titles[k][0] == 'P':

            plt.ylabel('\N{GREEK SMALL LETTER PHI} [rad]')
            plt.plot(time,inputFileCopy[k][1:].astype(float),plottypes[k%len(plottypes)-1])
        elif titles[k][0] == 'V':
            if max(inputFileCopy[k][1:].astype(float))<0.1:
                plt.ylabel('V [mV]')
                plt.plot(time,inputFileCopy[k][1:].astype(float)*1E6,plottypes[k%len(plottypes)-1])
            else:
                plt.ylabel('V [V]')
                plt.plot(time,inputFileCopy[k][1:].astype(float),plottypes[k%len(plottypes)-1])
        plt.xlabel('Time [ps]')
        legend.append(titles[k])
    plt.legend(legend[0:])
    plt.title(plottitle)
    plt.show()
elif plotmode==3: #separate
    for k in range(1,numberPlots+1):
        plt.figure(k)
        if titles[k][0] == 'I':
            plt.plot(time,inputFileCopy[k][1:].astype(float)*1e6) 
            plt.ylabel('I [\N{GREEK SMALL LETTER MU}A]')
        elif titles[k][0] == 'P':

            plt.ylabel('\N{GREEK SMALL LETTER PHI} [rad]')
            plt.plot(time,inputFileCopy[k][1:].astype(float)) 
        elif titles[k][0] == 'V':
            if max(inputFileCopy[k][1:].astype(float))<0.1:
                plt.ylabel('V [mV]')
                plt.plot(time,inputFileCopy[k][1:].astype(float)*1E6) 
            else:
                plt.ylabel('V [V]')
                plt.plot(time,inputFileCopy[k][1:].astype(float)) 
            #plt.plot(time,inputFileCopy[k][1:].astype(float)) 
        plt.xlabel('Time [ps]')
        plt.title(titles[k])
        plt.show()

       
         

