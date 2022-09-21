#generate inputs for testing neuron
#generate inputs for 3x3 input images
#uses voltage source instead of current source
#complete system
import numpy as np
import math
import os
import sys
import time

from matplotlib import pyplot as plt
AMPLITUDE = 1*1000e-6; #MICROAMPS
PULSE_WIDTH = 50; #PS #was 400ps, then 100ps
RISE_TIME = 5; #PS
FALL_TIME = 5; #PS
PERIOD = 200; #P #ps #was 2000 originally was 400 for fast synapse
PERIODFAST=math.trunc(PERIOD/2); #PS
DELAY =550; #PS 
CATCHUP=50; #maybe period over 4
#CATCHUP =PERIOD/4
#TRAININGTIME =25000000 #ps
TRAININGTIME =1280000 #ps
#for comparator
CONSTSET ='47U'
DELAYSAW=640;
PERIODSAW=50
AMPSAW = 670E6
RISESAW=25
FALLSAW=35

ENDINITIALTIME=2500000#ps
ENDINITIALTIME=2500#ps
epochs =40
initial1=10e-6
initial2=-80e-6
initial3=10e-6
initial4=0e-6
initial5=70e-6
initial6=-40e-6
initial7=0e-6
initial8=-30e-6
initial9=10e-6
initial10=10e-6
initial11=20e-6
initial12=30e-6
initial13=-60e-6
initial14=-40e-6
initial15=-0e-6
initial16=-10e-6
initial17=20e-6
initial18=90e-6
initialb1=0e-6 #was 20
initialb2=-0e-6
initialb2=0e-6 #was40
#try doing like 7 epochs and then do fir 1st neuron only
#AEIOU
#INPUTS
in1 = np.array([0,1,0,1])#1,1
in2 = np.array([0,0,1,1])
saw = np.array([1, 0, 0, 0])

#TARGETS
targetA = np.array([0,1,1,0])


#TILE EACH INPUT AND TARGET
in1new = np.tile(in1,epochs) #ALWAYS
in2new = np.tile(in2,epochs) #ALWAYS
# inbiasnew = np.tile(inbias,epochs) #ALWAYS
sawnew = np.tile(saw,epochs*10)

targetAnew = np.tile(targetA,epochs) #ALWAYS

length = (np.size(targetAnew))
np.set_printoptions(threshold=sys.maxsize)
#INDICES
in1_indices = np.where(in1new >0);
in2_indices = np.where(in2new >0);
# inbias_indices = np.where(inbiasnew >0);
targetA_indices = np.where(targetAnew >0);
saw_indices = np.where(sawnew>0);


#times
timesin1 = np.multiply(in1_indices,PERIOD)+DELAY;
timesin2 = np.multiply(in2_indices,PERIOD)+DELAY;
# timesinbias = np.multiply(inbias_indices,PERIOD)+DELAY;

timestargetA = np.multiply(targetA_indices,PERIOD)+DELAY;

timessaw = np.multiply(saw_indices,PERIODSAW)+DELAYSAW;


#waveform target
targetnewA = np.insert(timestargetA[0],np.where(timestargetA[0]>1)[0]+1,timestargetA[0][np.where(timestargetA[0]>1)[0]]+RISE_TIME)
#targetnewA_copy = np.copy(targetnewA)
targetnewA = np.insert(targetnewA,   np.where(targetnewA % 10 ==5)[0]      , 0)
targetnewA = np.insert(targetnewA,  np.where(targetnewA % 10 ==5)[0]+1    ,69)
targetnewA = np.insert(targetnewA,np.where(targetnewA==69)[0]+1, targetnewA[np.where(targetnewA==69)[0]-1]+PULSE_WIDTH)
targetnewA = np.insert(targetnewA,np.where(targetnewA==69)[0]+2, targetnewA[np.where(targetnewA==69)[0]-1]+PULSE_WIDTH+FALL_TIME)
targetnewA = np.insert(targetnewA,np.where(targetnewA==69)[0]+3,0)
targetnewA = np.insert(targetnewA,np.where(targetnewA==69)[0]+2,69)
timesfinaltargetA = np.multiply(targetnewA,1e-12)
timesfinaltargetA[timesfinaltargetA==69e-12] = AMPLITUDE*np.repeat(targetAnew[targetA_indices],2)
print(timesfinaltargetA)




#input1
in1new2 = np.insert(timesin1[0],np.where(timesin1[0]>1)[0]+1,timesin1[0][np.where(timesin1[0]>1)[0]]+RISE_TIME)
in1new2 = np.insert(in1new2,   np.where(in1new2 % 10 ==5)[0]      , 0)
in1new2 = np.insert(in1new2,  np.where(in1new2 % 10 ==5)[0]+1    ,69)
in1new2 = np.insert(in1new2,np.where(in1new2==69)[0]+1, in1new2[np.where(in1new2==69)[0]-1]+PULSE_WIDTH)
in1new2 = np.insert(in1new2,np.where(in1new2==69)[0]+2, in1new2[np.where(in1new2==69)[0]-1]+PULSE_WIDTH+FALL_TIME)
in1new2 = np.insert(in1new2,np.where(in1new2==69)[0]+3,0)
in1new2 = np.insert(in1new2,np.where(in1new2==69)[0]+2,69)
timesfinalin1 = np.multiply(in1new2,1e-12)
timesfinalin1[timesfinalin1==69e-12] = AMPLITUDE*np.repeat(in1new[in1_indices],2)
print('final 1',timesfinalin1)
#INPUT 2
in2new2 = np.insert(timesin2[0],np.where(timesin2[0]>1)[0]+1,timesin2[0][np.where(timesin2[0]>1)[0]]+RISE_TIME)
in2new2 = np.insert(in2new2,   np.where(in2new2 % 10 ==5)[0]      , 0)
in2new2 = np.insert(in2new2,  np.where(in2new2 % 10 ==5)[0]+1    ,69)
in2new2 = np.insert(in2new2,np.where(in2new2==69)[0]+1, in2new2[np.where(in2new2==69)[0]-1]+PULSE_WIDTH)
in2new2 = np.insert(in2new2,np.where(in2new2==69)[0]+2, in2new2[np.where(in2new2==69)[0]-1]+PULSE_WIDTH+FALL_TIME)
in2new2 = np.insert(in2new2,np.where(in2new2==69)[0]+3,0)
in2new2 = np.insert(in2new2,np.where(in2new2==69)[0]+2,69)
timesfinalin2 = np.multiply(in2new2,1e-12)
timesfinalin2[timesfinalin2==69e-12] = AMPLITUDE*np.repeat(in2new[in2_indices],2)

extra=np.array(0)
#formatted saw
sawnew2 = np.insert(timessaw[0],np.where(timessaw[0]>1)[0]+1,timessaw[0][np.where(timessaw[0]>1)[0]]+RISESAW)
#sawnew2.astype(np.float64)
#sawnew_copy = np.copy(sawnew)
sawnew2 = np.insert(sawnew2,   np.where(sawnew2 % 10 ==5)[0]      , 0)
sawnew2 = np.insert(sawnew2,  np.where(sawnew2 % 10 ==5)[0]+1    ,AMPSAW)
#sawnew2 = np.where(sawnew2==AMPSAW,69)[0]
#print(sawnew2)
#print(extra)
#sawnew2 = np.insert(sawnew2,  np.where(sawnew2 ==AMPSAW)[0] ,1)
#sawnew2 = np.delete(sawnew2,  np.where(sawnew2 ==AMPSAW)[0])
#sawnew2 = np.insert(sawnew2,np.where(sawnew2==AMPSAW)[0]+1, sawnew2[np.where(sawnew2==AMPSAW)[0]-1])
sawnew2 = np.insert(sawnew2,  np.where(sawnew2 ==AMPSAW)[0]+1    ,sawnew2[np.where(sawnew2 ==AMPSAW)[0]-1]+FALLSAW)
sawnew2 = np.insert(sawnew2,  np.where(sawnew2 ==AMPSAW)[0]+2    ,0)
print(sawnew2)
#sawnew2 = np.insert(sawnew2,np.where(sawnew2==AMPSAW)[0]+2, sawnew2[np.where(sawnew2==AMPSAW)[0]-1]+35)
#print(sawnew2)
#sawnew2 = np.insert(sawnew2,np.where(sawnew2==AMPSAW)[0]+3,0)

#sawnew2 = np.insert(sawnew2,np.where(sawnew2==69)[0]+2,69)
timesfinalsaw = np.multiply(sawnew2,1e-12)
#timesfinalsaw[timesfinalsaw==69e-12] = 670e-6*np.repeat(sawnew[saw_indices],2)


#FORMATTED TARGETS
formattedtargetA = str(timesfinaltargetA).strip('[')
formattedtargetA = formattedtargetA.strip(']')
formattedtargetA = formattedtargetA.replace('\n','\n+')

#print('ITARGET 0 TARGET1 '+ 'PWL(0 0 20P 0 '+formattedtarget+')')  


#FORMATTED INPUTS

formattedin1 = str(timesfinalin1).strip('[')
formattedin1 = formattedin1.strip(']')
formattedin1 = formattedin1.replace('\n','\n+')

formattedin2 = str(timesfinalin2).strip('[')
formattedin2 = formattedin2.strip(']')
formattedin2 = formattedin2.replace('\n','\n+')

print(timesfinalsaw)
formattedsaw = str(timesfinalsaw).strip('[')
#print(timesfinalsaw)
str(formattedsaw).replace('[','')

#print(formattedsaw)
formattedsaw = str(timesfinalsaw).strip(']')
#print(formattedsaw)
formattedsaw= formattedsaw.replace('\n','\n+')




endtime = max(timesfinaltargetA[len(timesfinaltargetA)-2],timesfinalin1[len(timesfinalin1)-2],timesfinalin2[len(timesfinalin2)-2]) + 1000E-12;
TRAININGTIME=timesfinaltargetA[-2]


#need to make a write file, and generate cmds for inputs as well
path = 'COMPONENTS/'

with open('inputs.cir', 'w') as f:
    f.write('***    THIS IS AN AUTOGENERATED FILE   ***\n');
    #INCLUDES
    f.write('***The Requested Inputs***')
    f.write('.include ' +path+'nand.cir \n.include ' +path+'or.cir\n.include ' +path+'and.cir\n.include ' +path+'and2.cir\n.include ' +path+'delay4.cir\n.include ' +path+'XOR.cir\n.include '+path+'ADJUSTNEW2.cir\n.include ' +path+'componentsedit.cir\n.include ' +path+'convinterface.cir\n.include ' +path+'LSmitll_bufft_v1p5.cir\n.include ' +path+'STOREedit.cir\n.include ' +path+'LSmitll_DCSFQ_PTLTX_v1p5.cir\n.include ' +path+'NEURON3.cir\n.include '+path+'COMP3.cir\n')
    f.write('.tran 1ps '+ str(endtime) +' 0ps 1p\n'); #SIMULATE
    f.write('***    INPUTS  ***\n') #SOURCES
#CURRENT
    f.write('IIN1 0 IN1 '+ 'PWL(0 0 20P 0 '+formattedin1+')\n')
    f.write('IIN2 0 IN2 '+ 'PWL(0 0 20P 0 '+formattedin2+')\n') 
    f.write('IINBIAS1 0 INB11 pulse(0 '+ str(AMPLITUDE) +' '+str(DELAY)+ 'p 5p 5p ' +str(PULSE_WIDTH)+'p '+str(PERIOD)+'p)\n' )
    f.write('ITARGET 0 TARGET0  '+ 'PWL(0 0 20P 0 '+formattedtargetA+')\n') 
    #f.write('Iactualsynbias21 0 ACTUALSYNB21x PWL'+'(0 0 ' +formattedsaw.replace('[','') + ')\n')


    #f.write('VTARGETA TARGET0A 0 '+ 'PWL(0 0 20P 0 '+formattedtargetA+')\n') 
    #f.write('RTARGETA TARGET0A TARGETA '+ '1\n') 


#print('SAW: '+ formattedsaw)
#print(sawnew2)
#print(timesfinalsaw)
#print(formattedsaw)
#print(680e-6)


#os.system('python3 josim-plot OUTPUTS/genneuron.csv -t stacked')

