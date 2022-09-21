#generate inputs for testing neuron
#generate inputs for 3x3 input images
#uses voltage source instead of current source
#complete system
import numpy as np
import math
import os
import sys
import time
#from data import *
from matplotlib import pyplot as plt
AMPLITUDE = 1000e-6; #MICROAMPS
PULSE_WIDTH = 50; #PS #was 400ps
RISE_TIME = 5; #PS
FALL_TIME = 5; #PS
PERIOD = 200; #P #ps #was 2000
PERIODFAST=math.trunc(PERIOD/2); #PS
DELAY =150; #PS
CATCHUP=50
#TRAININGTIME =25000000 #ps
TRAININGTIME =1280000 #ps
#for comparator - I THINK HIGHER MAKES IT HARDER TO SATURATE
CONSTSET1 =' 40U'
#for comparator - I THINK HIGHER MAKES IT HARDER TO SATURATE
CONSTSET2 ='100U'

ENDINITIALTIME=2500000#ps
epochs =2; 
# initial1=40e-6
# initial2=-10e-6
# initial3=10e-6
# initial4=0e-6
# initial5=70e-6
# initial6=-40e-6
# initial7=0e-6
# initial8=10e-6
# initial9=10e-6
# initial10=10e-6
# initial11=20e-6
# initial12=30e-6
# initial13=-60e-6
# initial14=-40e-6
# initial15=-0e-6
# initial16=-10e-6
# initial17=20e-6
# initial18=90e-6
# initialb1=20e-6
# initialb2=0e-6
# initialb2=40e-6

initial1=0
initial2=-0
initial3=0
initial4=0
initial5=0
initial6=-0
initial7=0
initial8=0
initial9=0
initial10=0
initial11=0
initial12=0
initial13=0
initial14=0
initial15=0
initial16=0
initial17=0
initial18=0
initialb1=0
initialb2=0
initialb2=0
#try doing like 7 epochs and then do fir 1st neuron only
#AEIOU
#INPUTS
in1 = np.array([1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1]); #1,1
in2 = np.array([1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1]); #1,2
in3 = np.array([1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1]); #1,3
in4 = np.array([1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1]); #2,1
in5 = np.array([1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1]); #2,2
in6 = np.array([1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0]); #2,3
in7 = np.array([1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1]); #3,1
in8 = np.array([0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1]); #3,2
in9 = np.array([1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1]); #3,3
#TARGETS
targetA = np.array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0])
targetU = np.array([0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0])
targetI = np.array([0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0])
targetE = np.array([0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1])
targetO = np.array([0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

#TILE EACH INPUT AND TARGET
in1new = np.tile(in1,epochs) #ALWAYS
in2new = np.tile(in2,epochs) #ALWAYS
in3new = np.tile(in3,epochs) #ALWAYS
in4new = np.tile(in4,epochs) #ALWAYS
in5new = np.tile(in5,epochs) #ALWAYS
in6new = np.tile(in6,epochs) #ALWAYS
in7new = np.tile(in7,epochs) #ALWAYS
in8new = np.tile(in8,epochs) #ALWAYS
in9new = np.tile(in9,epochs) #ALWAYS
targetAnew = np.tile(targetA,epochs) #ALWAYS
targetUnew = np.tile(targetU,epochs) #ALWAYS
targetInew = np.tile(targetI,epochs) #ALWAYS
targetEnew = np.tile(targetE,epochs) #ALWAYS
targetOnew = np.tile(targetO,epochs) #ALWAYS
length = (np.size(targetAnew))
np.set_printoptions(threshold=sys.maxsize)
#INDICES
in1_indices = np.where(in1new >0);
in2_indices = np.where(in2new >0);
in3_indices = np.where(in3new >0);
in4_indices = np.where(in4new >0);
in5_indices = np.where(in5new >0);
in6_indices = np.where(in6new >0);
in7_indices = np.where(in7new >0);
in8_indices = np.where(in8new >0);
in9_indices = np.where(in9new >0);
targetA_indices = np.where(targetAnew >0);
targetU_indices = np.where(targetUnew >0);
targetI_indices = np.where(targetInew >0);
targetE_indices = np.where(targetEnew >0);
targetO_indices = np.where(targetOnew >0);

#times
timesin1 = np.multiply(in1_indices,PERIOD)+DELAY;
timesin2 = np.multiply(in2_indices,PERIOD)+DELAY;
timesin3 = np.multiply(in3_indices,PERIOD)+DELAY;
timesin4 = np.multiply(in4_indices,PERIOD)+DELAY;
timesin5 = np.multiply(in5_indices,PERIOD)+DELAY;
timesin6 = np.multiply(in6_indices,PERIOD)+DELAY;
timesin7 = np.multiply(in7_indices,PERIOD)+DELAY;
timesin8 = np.multiply(in8_indices,PERIOD)+DELAY;
timesin9 = np.multiply(in9_indices,PERIOD)+DELAY;


timestargetA = np.multiply(targetA_indices,PERIOD)+DELAY;
timestargetU = np.multiply(targetU_indices,PERIOD)+DELAY;
timestargetI = np.multiply(targetI_indices,PERIOD)+DELAY;
timestargetE = np.multiply(targetE_indices,PERIOD)+DELAY;
timestargetO = np.multiply(targetO_indices,PERIOD)+DELAY;

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
timesfinaltargetA[timesfinaltargetA==69e-12] = 1e-3*np.repeat(targetAnew[targetA_indices],2)
print(timesfinaltargetA)


targetnewU = np.insert(timestargetU[0],np.where(timestargetU[0]>1)[0]+1,timestargetU[0][np.where(timestargetU[0]>1)[0]]+RISE_TIME)
#targetnewU_copy = np.copy(targetnewU)
targetnewU = np.insert(targetnewU,   np.where(targetnewU % 10 ==5)[0]      , 0)
targetnewU = np.insert(targetnewU,  np.where(targetnewU % 10 ==5)[0]+1    ,69)
targetnewU = np.insert(targetnewU,np.where(targetnewU==69)[0]+1, targetnewU[np.where(targetnewU==69)[0]-1]+PULSE_WIDTH)
targetnewU = np.insert(targetnewU,np.where(targetnewU==69)[0]+2, targetnewU[np.where(targetnewU==69)[0]-1]+PULSE_WIDTH+FALL_TIME)
targetnewU = np.insert(targetnewU,np.where(targetnewU==69)[0]+3,0)
targetnewU = np.insert(targetnewU,np.where(targetnewU==69)[0]+2,69)
timesfinaltargetU = np.multiply(targetnewU,1e-12)
timesfinaltargetU[timesfinaltargetU==69e-12] = 1e-3*np.repeat(targetUnew[targetU_indices],2)

targetnewI = np.insert(timestargetI[0],np.where(timestargetI[0]>1)[0]+1,timestargetI[0][np.where(timestargetI[0]>1)[0]]+RISE_TIME)
#targetnewU_copy = np.copy(targetnewU)
targetnewI = np.insert(targetnewI,   np.where(targetnewI % 10 ==5)[0]      , 0)
targetnewI = np.insert(targetnewI,  np.where(targetnewI % 10 ==5)[0]+1    ,69)
targetnewI = np.insert(targetnewI,np.where(targetnewI==69)[0]+1, targetnewI[np.where(targetnewI==69)[0]-1]+PULSE_WIDTH)
targetnewI = np.insert(targetnewI,np.where(targetnewI==69)[0]+2, targetnewI[np.where(targetnewI==69)[0]-1]+PULSE_WIDTH+FALL_TIME)
targetnewI = np.insert(targetnewI,np.where(targetnewI==69)[0]+3,0)
targetnewI = np.insert(targetnewI,np.where(targetnewI==69)[0]+2,69)
timesfinaltargetI = np.multiply(targetnewI,1e-12)
timesfinaltargetI[timesfinaltargetI==69e-12] = 1e-3*np.repeat(targetInew[targetI_indices],2)

targetnewE = np.insert(timestargetE[0],np.where(timestargetE[0]>1)[0]+1,timestargetE[0][np.where(timestargetE[0]>1)[0]]+RISE_TIME)
#targetnewU_copy = np.copy(targetnewU)
targetnewE = np.insert(targetnewE,   np.where(targetnewE % 10 ==5)[0]      , 0)
targetnewE = np.insert(targetnewE,  np.where(targetnewE % 10 ==5)[0]+1    ,69)
targetnewE = np.insert(targetnewE,np.where(targetnewE==69)[0]+1, targetnewE[np.where(targetnewE==69)[0]-1]+PULSE_WIDTH)
targetnewE = np.insert(targetnewE,np.where(targetnewE==69)[0]+2, targetnewE[np.where(targetnewE==69)[0]-1]+PULSE_WIDTH+FALL_TIME)
targetnewE = np.insert(targetnewE,np.where(targetnewE==69)[0]+3,0)
targetnewE = np.insert(targetnewE,np.where(targetnewE==69)[0]+2,69)
timesfinaltargetE = np.multiply(targetnewE,1e-12)
timesfinaltargetE[timesfinaltargetE==69e-12] = 1e-3*np.repeat(targetEnew[targetE_indices],2)


targetnewO = np.insert(timestargetO[0],np.where(timestargetO[0]>1)[0]+1,timestargetO[0][np.where(timestargetO[0]>1)[0]]+RISE_TIME)
#targetnewU_copy = np.copy(targetnewU)
targetnewO = np.insert(targetnewO,   np.where(targetnewO % 10 ==5)[0]      , 0)
targetnewO = np.insert(targetnewO,  np.where(targetnewO % 10 ==5)[0]+1    ,69)
targetnewO = np.insert(targetnewO,np.where(targetnewO==69)[0]+1, targetnewO[np.where(targetnewO==69)[0]-1]+PULSE_WIDTH)
targetnewO = np.insert(targetnewO,np.where(targetnewO==69)[0]+2, targetnewO[np.where(targetnewO==69)[0]-1]+PULSE_WIDTH+FALL_TIME)
targetnewO = np.insert(targetnewO,np.where(targetnewO==69)[0]+3,0)
targetnewO = np.insert(targetnewO,np.where(targetnewO==69)[0]+2,69)
timesfinaltargetO = np.multiply(targetnewO,1e-12)
timesfinaltargetO[timesfinaltargetO==69e-12] = 1e-3*np.repeat(targetOnew[targetO_indices],2)



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

#INPUT 3
in3new2 = np.insert(timesin3[0],np.where(timesin3[0]>1)[0]+1,timesin3[0][np.where(timesin3[0]>1)[0]]+RISE_TIME)
in3new2 = np.insert(in3new2,   np.where(in3new2 % 10 ==5)[0]      , 0)
in3new2 = np.insert(in3new2,  np.where(in3new2 % 10 ==5)[0]+1    ,69)
in3new2 = np.insert(in3new2,np.where(in3new2==69)[0]+1, in3new2[np.where(in3new2==69)[0]-1]+PULSE_WIDTH)
in3new2 = np.insert(in3new2,np.where(in3new2==69)[0]+2, in3new2[np.where(in3new2==69)[0]-1]+PULSE_WIDTH+FALL_TIME)
in3new2 = np.insert(in3new2,np.where(in3new2==69)[0]+3,0)
in3new2 = np.insert(in3new2,np.where(in3new2==69)[0]+2,69)
timesfinalin3 = np.multiply(in3new2,1e-12)
timesfinalin3[timesfinalin3==69e-12] = AMPLITUDE*np.repeat(in3new[in3_indices],2)

#INPUT 4
in4new2 = np.insert(timesin4[0],np.where(timesin4[0]>1)[0]+1,timesin4[0][np.where(timesin4[0]>1)[0]]+RISE_TIME)
in4new2 = np.insert(in4new2,   np.where(in4new2 % 10 ==5)[0]      , 0)
in4new2 = np.insert(in4new2,  np.where(in4new2 % 10 ==5)[0]+1    ,69)
in4new2 = np.insert(in4new2,np.where(in4new2==69)[0]+1, in4new2[np.where(in4new2==69)[0]-1]+PULSE_WIDTH)
in4new2 = np.insert(in4new2,np.where(in4new2==69)[0]+2, in4new2[np.where(in4new2==69)[0]-1]+PULSE_WIDTH+FALL_TIME)
in4new2 = np.insert(in4new2,np.where(in4new2==69)[0]+3,0)
in4new2 = np.insert(in4new2,np.where(in4new2==69)[0]+2,69)
timesfinalin4 = np.multiply(in4new2,1e-12)
timesfinalin4[timesfinalin4==69e-12] = AMPLITUDE*np.repeat(in4new[in4_indices],2)
#INPUT 5
in5new2 = np.insert(timesin5[0],np.where(timesin5[0]>1)[0]+1,timesin5[0][np.where(timesin5[0]>1)[0]]+RISE_TIME)
in5new2 = np.insert(in5new2,   np.where(in5new2 % 10 ==5)[0]      , 0)
in5new2 = np.insert(in5new2,  np.where(in5new2 % 10 ==5)[0]+1    ,69)
in5new2 = np.insert(in5new2,np.where(in5new2==69)[0]+1, in5new2[np.where(in5new2==69)[0]-1]+PULSE_WIDTH)
in5new2 = np.insert(in5new2,np.where(in5new2==69)[0]+2, in5new2[np.where(in5new2==69)[0]-1]+PULSE_WIDTH+FALL_TIME)
in5new2 = np.insert(in5new2,np.where(in5new2==69)[0]+3,0)
in5new2 = np.insert(in5new2,np.where(in5new2==69)[0]+2,69)
timesfinalin5 = np.multiply(in5new2,1e-12)
timesfinalin5[timesfinalin5==69e-12] = AMPLITUDE*np.repeat(in5new[in5_indices],2)
#INPUT 6
in6new2 = np.insert(timesin6[0],np.where(timesin6[0]>1)[0]+1,timesin6[0][np.where(timesin6[0]>1)[0]]+RISE_TIME)
in6new2 = np.insert(in6new2,   np.where(in6new2 % 10 ==5)[0]      , 0)
in6new2 = np.insert(in6new2,  np.where(in6new2 % 10 ==5)[0]+1    ,69)
in6new2 = np.insert(in6new2,np.where(in6new2==69)[0]+1, in6new2[np.where(in6new2==69)[0]-1]+PULSE_WIDTH)
in6new2 = np.insert(in6new2,np.where(in6new2==69)[0]+2, in6new2[np.where(in6new2==69)[0]-1]+PULSE_WIDTH+FALL_TIME)
in6new2 = np.insert(in6new2,np.where(in6new2==69)[0]+3,0)
in6new2 = np.insert(in6new2,np.where(in6new2==69)[0]+2,69)
timesfinalin6 = np.multiply(in6new2,1e-12)
timesfinalin6[timesfinalin6==69e-12] = AMPLITUDE*np.repeat(in6new[in6_indices],2)
#INPUT 7
in7new2 = np.insert(timesin7[0],np.where(timesin7[0]>1)[0]+1,timesin7[0][np.where(timesin7[0]>1)[0]]+RISE_TIME)
in7new2 = np.insert(in7new2,   np.where(in7new2 % 10 ==5)[0]      , 0)
in7new2 = np.insert(in7new2,  np.where(in7new2 % 10 ==5)[0]+1    ,69)
in7new2 = np.insert(in7new2,np.where(in7new2==69)[0]+1, in7new2[np.where(in7new2==69)[0]-1]+PULSE_WIDTH)
in7new2 = np.insert(in7new2,np.where(in7new2==69)[0]+2, in7new2[np.where(in7new2==69)[0]-1]+PULSE_WIDTH+FALL_TIME)
in7new2 = np.insert(in7new2,np.where(in7new2==69)[0]+3,0)
in7new2 = np.insert(in7new2,np.where(in7new2==69)[0]+2,69)
timesfinalin7 = np.multiply(in7new2,1e-12)
timesfinalin7[timesfinalin7==69e-12] = AMPLITUDE*np.repeat(in7new[in7_indices],2)
#INPUT 8
in8new2 = np.insert(timesin8[0],np.where(timesin8[0]>1)[0]+1,timesin8[0][np.where(timesin8[0]>1)[0]]+RISE_TIME)
in8new2 = np.insert(in8new2,   np.where(in8new2 % 10 ==5)[0]      , 0)
in8new2 = np.insert(in8new2,  np.where(in8new2 % 10 ==5)[0]+1    ,69)
in8new2 = np.insert(in8new2,np.where(in8new2==69)[0]+1, in8new2[np.where(in8new2==69)[0]-1]+PULSE_WIDTH)
in8new2 = np.insert(in8new2,np.where(in8new2==69)[0]+2, in8new2[np.where(in8new2==69)[0]-1]+PULSE_WIDTH+FALL_TIME)
in8new2 = np.insert(in8new2,np.where(in8new2==69)[0]+3,0)
in8new2 = np.insert(in8new2,np.where(in8new2==69)[0]+2,69)
timesfinalin8 = np.multiply(in8new2,1e-12)
timesfinalin8[timesfinalin8==69e-12] = AMPLITUDE*np.repeat(in3new[in8_indices],2)
#INPUT 9
in9new2 = np.insert(timesin9[0],np.where(timesin9[0]>1)[0]+1,timesin9[0][np.where(timesin9[0]>1)[0]]+RISE_TIME)
in9new2 = np.insert(in9new2,   np.where(in9new2 % 10 ==5)[0]      , 0)
in9new2 = np.insert(in9new2,  np.where(in9new2 % 10 ==5)[0]+1    ,69)
in9new2 = np.insert(in9new2,np.where(in9new2==69)[0]+1, in9new2[np.where(in9new2==69)[0]-1]+PULSE_WIDTH)
in9new2 = np.insert(in9new2,np.where(in9new2==69)[0]+2, in9new2[np.where(in9new2==69)[0]-1]+PULSE_WIDTH+FALL_TIME)
in9new2 = np.insert(in9new2,np.where(in9new2==69)[0]+3,0)
in9new2 = np.insert(in9new2,np.where(in9new2==69)[0]+2,69)
timesfinalin9 = np.multiply(in9new2,1e-12)
timesfinalin9[timesfinalin9==69e-12] = AMPLITUDE*np.repeat(in9new[in9_indices],2)


#FORMATTED TARGETS
formattedtargetA = str(timesfinaltargetA).strip('[')
formattedtargetA = formattedtargetA.strip(']')
formattedtargetA = formattedtargetA.replace('\n','\n+')

#print('ITARGET 0 TARGET1 '+ 'PWL(0 0 20P 0 '+formattedtarget+')')  

formattedtargetU = str(timesfinaltargetU).strip('[')
formattedtargetU = formattedtargetU.strip(']')
formattedtargetU = formattedtargetU.replace('\n','\n+')
#print('ITARGET2 0 TARGET2 '+ 'PWL(0 0 20P 0 '+formattedtarget2+')')  

formattedtargetI = str(timesfinaltargetI).strip('[')
formattedtargetI = formattedtargetI.strip(']')
formattedtargetI = formattedtargetI.replace('\n','\n+')

formattedtargetE = str(timesfinaltargetE).strip('[')
formattedtargetE = formattedtargetE.strip(']')
formattedtargetE = formattedtargetE.replace('\n','\n+')

formattedtargetO = str(timesfinaltargetO).strip('[')
formattedtargetO = formattedtargetO.strip(']')
formattedtargetO = formattedtargetO.replace('\n','\n+')

#FORMATTED INPUTS

formattedin1 = str(timesfinalin1).strip('[')
formattedin1 = formattedin1.strip(']')
formattedin1 = formattedin1.replace('\n','\n+')

formattedin2 = str(timesfinalin2).strip('[')
formattedin2 = formattedin2.strip(']')
formattedin2 = formattedin2.replace('\n','\n+')

formattedin3 = str(timesfinalin3).strip('[')
formattedin3 = formattedin3.strip(']')
formattedin3 = formattedin3.replace('\n','\n+')

formattedin4 = str(timesfinalin4).strip('[')
formattedin4 = formattedin4.strip(']')
formattedin4 = formattedin4.replace('\n','\n+')

formattedin5 = str(timesfinalin5).strip('[')
formattedin5 = formattedin5.strip(']')
formattedin5 = formattedin5.replace('\n','\n+')

formattedin6 = str(timesfinalin6).strip('[')
formattedin6 = formattedin6.strip(']')
formattedin6 = formattedin6.replace('\n','\n+')

formattedin7 = str(timesfinalin7).strip('[')
formattedin7 = formattedin7.strip(']')
formattedin7 = formattedin7.replace('\n','\n+')

formattedin8 = str(timesfinalin8).strip('[')
formattedin8 = formattedin8.strip(']')
formattedin8 = formattedin8.replace('\n','\n+')

formattedin9 = str(timesfinalin9).strip('[')
formattedin9 = formattedin9.strip(']')
formattedin9 = formattedin9.replace('\n','\n+')

endtime = max(timesfinaltargetA[len(timesfinaltargetA)-2],timesfinaltargetU[len(timesfinaltargetU)-2],timesfinalin1[len(timesfinalin1)-2],timesfinalin2[len(timesfinalin2)-2],timesfinalin3[len(timesfinalin3)-2],timesfinalin4[len(timesfinalin4)-2],timesfinalin5[len(timesfinalin5)-2],timesfinalin5[len(timesfinalin5)-2],timesfinalin6[len(timesfinalin6)-2],timesfinalin7[len(timesfinalin7)-2],timesfinalin8[len(timesfinalin8)-2],timesfinalin9[len(timesfinalin9)-2]) + 1000E-12;



#need to make a write file, and generate cmds for inputs as well
path = 'COMPONENTS/'

with open('genneuron.cir', 'w') as f:
    f.write('***    THIS IS AN AUTOGENERATED FILE   ***\n');
    #f.write('.include ' +path+'nand.cir \n.include ' +path+'or.cir\n.include ' +path+'and.cir\n.include ' +path+'and2.cir\n.include ' +path+'delay4.cir\n.include ' +path+'XOR.cir\n.include '+path+'ADJUSTNEW2.cir\n.include ' +path+'componentsedit.cir\n.include ' +path+'convinterface.cir\n.include ' +path+'LSmitll_bufft_v1p5.cir\n.include ' +path+'STOREedit.cir\n.include ' +path+'LSmitll_DCSFQ_PTLTX_v1p5.cir\n.include ' +path+'NEURON11.cir\n.include '+path+'COMP11.cir\n')
    f.write('.include LSmitll_bufft_v1p5.cir \n.include LSmitll_PTLRX_SFQDC_v1p5.cir \n.include LSmitll_SPLITT_v1p5.cir\n.INCLUDE LSMITLL_JTLT_V1P5.cir\n.INCLUDE LSMITLL_MERGET_V1P5.cir\n.include storeedit.cir\n.INCLUDE COMPONENTSEDIT.cir\n.INCLUDE COMP3.cir\n.INCLUDE COMP4.cir\n.INCLUDE COMP5.cir\n.INCLUDE COMP6.cir\n.include conv.cir\n.INCLUDE synapsenext2.cir\n.INCLUDE DCPULSER.cir\n.INCLUDE MULTISPLIT.cir\n.include transmit.cir\n.INCLUDE DELAY7.cir\n.INCLUDE AND.cir\n.INCLUDE AND3.cir\n.INCLUDE OR3.cir\n.include PERCEPTRON.cir\n.include PERCEPTRON2.cir\n.include CONVINTERFACE.cir\n.include NEURONLETTERS.cir\n')
    f.write('.tran 1ps '+ str(endtime) +' 0ps 1p\n'); #change back to 0.2ps maybe
    #f.write('.tran 1ps '+ '50000p' +' 0ps 1p\n'); #change back to 0.2ps maybe

    f.write('***    INPUTS  ***\n')
    #either do voltage source or increase current amplitude in formatting maybe
    f.write('IIN1 0 IIN1x '+ 'PWL(0 0 20P 0 '+formattedin1+')\n')
    f.write('IIN2 0 IIN2x  '+ 'PWL(0 0 20P 0 '+formattedin2+')\n') 
    f.write('IIN3 0 IIN3x  '+ 'PWL(0 0 20P 0 '+formattedin3+')\n')
    f.write('IIN4 0 IIN4x  '+ 'PWL(0 0 20P 0 '+formattedin4+')\n') 
    f.write('IIN5 0 IIN5x  '+ 'PWL(0 0 20P 0 '+formattedin5+')\n')
    f.write('IIN6 0 IIN6x  '+ 'PWL(0 0 20P 0 '+formattedin6+')\n') 
    f.write('IIN7 0 IIN7x  '+ 'PWL(0 0 20P 0 '+formattedin7+')\n')
    f.write('IIN8 0 IIN8x  '+ 'PWL(0 0 20P 0 '+formattedin8+')\n') 
    f.write('IIN9 0 IIN9x '+ 'PWL(0 0 20P 0 '+formattedin9+')\n')
    #f.write('IIN10 0 IIN10x '+ 'PWL(0 0 20P 0 '+formattedin1+')\n')
    # f.write('IIN11 0 IIN11x  '+ 'PWL(0 0 20P 0 '+formattedin2+')\n') 
    # f.write('IIN12 0 IIN12x  '+ 'PWL(0 0 20P 0 '+formattedin3+')\n')
    # f.write('IIN13 0 IIN13x  '+ 'PWL(0 0 20P 0 '+formattedin4+')\n') 
    # f.write('IIN14 0 IIN14x  '+ 'PWL(0 0 20P 0 '+formattedin5+')\n')
    # f.write('IIN15 0 IIN15x  '+ 'PWL(0 0 20P 0 '+formattedin6+')\n') 
    # f.write('IIN16 0 IIN16x  '+ 'PWL(0 0 20P 0 '+formattedin7+')\n')
    # f.write('IIN17 0 IIN17x  '+ 'PWL(0 0 20P 0 '+formattedin8+')\n') 
    # f.write('IIN18 0 IIN18x '+ 'PWL(0 0 20P 0 '+formattedin9+')\n')
    f.write('LIN1 IIN1x IN1 '+ '1p\n')
    f.write('LIN2 IIN2x IN2 '+ '1p\n')
    f.write('LIN3 IIN3x IN3 '+ '1p\n')
    f.write('LIN4 IIN4x IN4 '+ '1p\n') 
    f.write('LIN5 IIN5x IN5 '+ '1p\n')
    f.write('LIN6 IIN6x IN6 '+ '1p\n')
    f.write('LIN7 IIN7x IN7 '+ '1p\n')
    f.write('LIN8 IIN8x IN8 '+ '1p\n')
    f.write('LIN9 IIN9x IN9 '+ '1p\n') 
    #f.write('LIN10 IIN1x IN10 '+ '1p\n')
    #f.write('LIN11 IIN2x IN11 '+ '1p\n')
    #f.write('LIN12 IIN3x IN12 '+ '1p\n')
    #f.write('LIN13 IIN4x IN13 '+ '1p\n') 
    #f.write('LIN14 IIN5x IN14 '+ '1p\n')
    #f.write('LIN15 IIN6x IN15 '+ '1p\n')
    #f.write('LIN16 IIN7x IN16 '+ '1p\n')
    #f.write('LIN17 IIN8x IN17 '+ '1p\n')
    #f.write('LIN18 IIN9x IN18 '+ '1p\n') 
    # f.write('LIN10 IIN1 IN10 '+ '1p\n')
    # f.write('LIN11 IIN2 IN11 '+ '1p\n')
    # f.write('LIN12 IIN3 IN12 '+ '1p\n')
    # f.write('LIN13 IIN4 IN13 '+ '1p\n') 
    # f.write('LIN14 IIN5 IN14 '+ '1p\n')
    # f.write('LIN15 IIN6 IN15 '+ '1p\n')
    # f.write('LIN16 IIN7 IN16 '+ '1p\n')
    # f.write('LIN17 IIN8 In17 '+ '1p\n')
    # f.write('LIN18 IIN9 IN18 '+ '1p\n')
    # f.write('LIN19 IIN1 IN19 '+ '1p\n')  
    # f.write('LIN20 IIN2 IN20 '+ '1p\n')
    # f.write('LIN21 IIN3 IN21 '+ '1p\n')
    # f.write('LIN22 IIN4 IN22 '+ '1p\n')
    # f.write('LIN23 IIN5 IN23 '+ '1p\n') 
    # f.write('LIN24 IIN6 IN24 '+ '1p\n')
    # f.write('LIN25 IIN7 IN25 '+ '1p\n')
    # f.write('LIN26 IIN8 IN26 '+ '1p\n')
    # f.write('LIN27 IIN9 In27 '+ '1p\n')
    # f.write('LIN28 IIN1 IN28 '+ '1p\n')
    # f.write('LIN29 IIN2 IN29 '+ '1p\n')  
    # f.write('LIN30 IIN3 IN30 '+ '1p\n')
    # f.write('LIN31 IIN4 IN31 '+ '1p\n')
    # f.write('LIN32 IIN5 IN32 '+ '1p\n')
    # f.write('LIN33 IIN6 IN33 '+ '1p\n') 
    # f.write('LIN34 IIN7 IN34 '+ '1p\n')
    # f.write('LIN35 IIN8 IN35 '+ '1p\n')
    # f.write('LIN36 IIN9 IN36 '+ '1p\n')
    # f.write('LIN37 IIN1 In37 '+ '1p\n')
    # f.write('LIN38 IIN2 IN38 '+ '1p\n')
    # f.write('LIN39 IIN3 IN39 '+ '1p\n')  
    # f.write('LIN40 IIN4 IN40 '+ '1p\n')
    # f.write('LIN41 IIN5 IN41 '+ '1p\n')
    # f.write('LIN42 IIN6 IN42 '+ '1p\n')
    # f.write('LIN43 IIN7 IN43 '+ '1p\n') 
    # f.write('LIN44 IIN8 IN44 '+ '1p\n')
    # f.write('LIN45 IIN9 IN45 '+ '1p\n')
  

    f.write('ITARGETA 0 TARGETA '+ 'PWL(0 0 20P 0 '+formattedtargetA+')\n') 
    #f.write('ITARGETU 0 TARGETU '+ 'PWL(0 0 20P 0 '+formattedtargetU+')\n') 
    #f.write('ITARGETE 0 TARGETE '+ 'PWL(0 0 20P 0 '+formattedtargetE+')\n') 
    #f.write('ITARGETI 0 TARGETI '+ 'PWL(0 0 20P 0 '+formattedtargetI+')\n') 
    #f.write('ITARGETO 0 TARGETO '+ 'PWL(0 0 20P 0 '+formattedtargetO+')\n') 

 
    f.write('IINBIAS1 0 INB01 pulse(0 1m '+str(DELAY)+ 'p 5p 5p '+ str(PULSE_WIDTH) +'p ' + str(PERIOD)+'p)\n' )
    f.write('LINBIAS1 INB01 INB1 '+ '1p\n') 
    #f.write('LINBIAS2 INB01 INB2 '+ '1p\n') 
    #f.write('IINBIAS2 0 INB02 pulse(0 1m '+str(DELAY)+ 'p 5p 5p '+ str(PULSE_WIDTH) +'p ' + str(PERIOD)+'p)\n' )
    #f.write('LINBIAS2 INB02 INB2 '+ '1p\n') 
    # f.write('IINBIAS3 0 INB03 0 pulse(0 1m '+str(DELAY)+ 'p 5p 5p '+ str(PULSE_WIDTH) +'p ' + str(PERIOD)+'p)\n' )
    # f.write('LINBIAS3 INB03 INB3 '+ '1p\n') 
    # f.write('IINBIAS4 0 INB04 0 pulse(0 1m '+str(DELAY)+ 'p 5p 5p '+ str(PULSE_WIDTH) +'p ' + str(PERIOD)+'p)\n' )
    # f.write('LINBIAS4 INB04 INB4 '+ '1p\n') 
    # f.write('IINBIAS5 0 INB05 0 pulse(0 1m '+str(DELAY)+ 'p 5p 5p '+ str(PULSE_WIDTH) +'p ' + str(PERIOD)+'p)\n' )
    # f.write('LINBIAS5 INB05 INB5 '+ '1p\n') 
    #changed bias pulse period
    #setup circuitry
    f.write('***   SETUP ***\n')
    f.write('VAC1   A1   0   SIN(0 723mV 10GHz '+str(DELAY+CATCHUP) +'ps 0)\nRAC1   A1   A2   1000\nLAC1   A2   A3   0.1p\nVAC2   B1   0   SIN(0 723mV 10GHz '+str(DELAY-(PERIOD/2/4)+CATCHUP)+'ps 0)\nRAC2   B1   B2   1000\nLAC2   B2   B3   0.1p\nVDC    DC1   0   pwl(0 0 20p 1023mV)\nRDC    DC1   DC2   1000\nLDC    DC2   DC3  0.1p\n')

    #f.write('VAC1   A1   0   SIN(0 723mV 1GHz 500ps 0)\nRAC1   A1   A2   1000\nLAC1   A2   A3   0.1p\nVAC2   B1   0   SIN(0 723mV 1GHz 250ps 0)\nRAC2   B1   B2   1000\nLAC2   B2   B3   0.1p\nVDC    DC1   0   pwl(0 0 20p 1023mV)\nRDC    DC1   DC2   1000\nLDC    DC2   DC3  0.1p\n')
    f.write('VDCconv    DCc1   0   PWL(0ps 0mV 20ps 1023mV '+ str(TRAININGTIME) +'ps 1023mV '+ str(TRAININGTIME+1) + 'ps 0)\nRDCconv    DCc1   DCc2  750\nLDCconv    DCc2   DCc3   0.1p\n')
    #f.write('VCONST    CONST   0   PWL(0ps 0mV 20ps 5mV 21ps 5mV )\nLCONST   CONST   CONST1  0.1P\n')
    f.write('***   INITIALIZE ***\n')
    f.write('IINITIAL1 0 INITIAL1 PWL(0 0 20P ' +str(initial1)+  '*-2*22.6u'+ ' '+ str(ENDINITIALTIME)+'p '+str(initial1)+ '*-2*22.6u'+' '+str(ENDINITIALTIME)+'p 0)\n')
    f.write('IINITIAL2 0 INITIAL2 PWL(0 0 20P ' +str(initial2)+ '*-2*22.6u'+ ' '+ str(ENDINITIALTIME)+'p '+str(initial2)+ '*-2*22.6u'+' '+str(ENDINITIALTIME)+'p 0)\n')
    f.write('IINITIAL3 0 INITIAL3 PWL(0 0 20P ' +str(initial3)+ '*-2*22.6u'+ ' '+ str(ENDINITIALTIME)+'p '+str(initial3)+ '*-2*22.6u'+' '+str(ENDINITIALTIME)+'p 0)\n')
    f.write('IINITIAL4 0 INITIAL4 PWL(0 0 20P ' +str(initial4)+ '*-2*22.6u'+ ' '+ str(ENDINITIALTIME)+'p '+str(initial4)+ '*-2*22.6u'+' '+str(ENDINITIALTIME)+'p 0)\n')
    f.write('IINITIAL5 0 INITIAL5 PWL(0 0 20P ' +str(initial5)+ '*-2*22.6u'+ ' '+ str(ENDINITIALTIME)+'p '+str(initial5)+ '*-2*22.6u'+' '+str(ENDINITIALTIME)+'p 0)\n')
    f.write('IINITIAL6 0 INITIAL6 PWL(0 0 20P ' +str(initial6)+ '*-2*22.6u'+ ' '+ str(ENDINITIALTIME)+'p '+str(initial6)+ '*-2*22.6u'+' '+str(ENDINITIALTIME)+'p 0)\n')
    f.write('IINITIAL7 0 INITIAL7 PWL(0 0 20P ' +str(initial7)+ '*-2*22.6u'+ ' '+ str(ENDINITIALTIME)+'p '+str(initial7)+ '*-2*22.6u'+' '+str(ENDINITIALTIME)+'p 0)\n')
    f.write('IINITIAL8 0 INITIAL8 PWL(0 0 20P ' +str(initial8)+ '*-2*22.6u'+ ' '+ str(ENDINITIALTIME)+'p '+str(initial8)+ '*-2*22.6u'+' '+str(ENDINITIALTIME)+'p 0)\n')
    f.write('IINITIAL9 0 INITIAL9 PWL(0 0 20P ' +str(initial9)+ '*-2*22.6u'+ ' '+ str(ENDINITIALTIME)+'p '+str(initial9)+ '*-2*22.6u'+' '+str(ENDINITIALTIME)+'p 0)\n')
    #f.write('IINITIAL10 0 INITIAL10 PWL(0 0 20P ' +str(initial10)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial10)+' '+str(ENDINITIALTIME)+'p 0)\n')
    #f.write('IINITIAL11 0 INITIAL11 PWL(0 0 20P ' +str(initial11)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial11)+' '+str(ENDINITIALTIME)+'p 0)\n')
    #f.write('IINITIAL12 0 INITIAL12 PWL(0 0 20P ' +str(initial12)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial12)+' '+str(ENDINITIALTIME)+'p 0)\n')
    #f.write('IINITIAL13 0 INITIAL13 PWL(0 0 20P ' +str(initial13)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial13)+' '+str(ENDINITIALTIME)+'p 0)\n')
    #f.write('IINITIAL14 0 INITIAL14 PWL(0 0 20P ' +str(initial14)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial14)+' '+str(ENDINITIALTIME)+'p 0)\n')
    #f.write('IINITIAL15 0 INITIAL15 PWL(0 0 20P ' +str(initial15)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial15)+' '+str(ENDINITIALTIME)+'p 0)\n')
    #f.write('IINITIAL16 0 INITIAL16 PWL(0 0 20P ' +str(initial16)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial16)+' '+str(ENDINITIALTIME)+'p 0)\n')
    #f.write('IINITIAL17 0 INITIAL17 PWL(0 0 20P ' +str(initial17)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial17)+' '+str(ENDINITIALTIME)+'p 0)\n')
    #f.write('IINITIAL18 0 INITIAL18 PWL(0 0 20P ' +str(initial18)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial18)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL19 0 INITIAL19 PWL(0 0 20P ' +str(initial9)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial9)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL20 0 INITIAL20 PWL(0 0 20P ' +str(initial10)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial10)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL21 0 INITIAL21 PWL(0 0 20P ' +str(initial11)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial11)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL22 0 INITIAL22 PWL(0 0 20P ' +str(initial12)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial12)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL23 0 INITIAL23 PWL(0 0 20P ' +str(initial13)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial13)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL24 0 INITIAL24 PWL(0 0 20P ' +str(initial14)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial14)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL25 0 INITIAL25 PWL(0 0 20P ' +str(initial15)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial15)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL26 0 INITIAL26 PWL(0 0 20P ' +str(initial16)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial16)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL27 0 INITIAL27 PWL(0 0 20P ' +str(initial17)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial17)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL28 0 INITIAL28 PWL(0 0 20P ' +str(initial9)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial9)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL29 0 INITIAL29 PWL(0 0 20P ' +str(initial1)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial10)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL30 0 INITIAL30 PWL(0 0 20P ' +str(initial11)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial11)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL31 0 INITIAL31 PWL(0 0 20P ' +str(initial14)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial14)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL32 0 INITIAL32 PWL(0 0 20P ' +str(initial3)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial13)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL33 0 INITIAL33 PWL(0 0 20P ' +str(initial15)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial15)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL34 0 INITIAL34 PWL(0 0 20P ' +str(initial3)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial13)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL35 0 INITIAL35 PWL(0 0 20P ' +str(initial16)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial16)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL36 0 INITIAL36 PWL(0 0 20P ' +str(initial11)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial11)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL37 0 INITIAL37 PWL(0 0 20P ' +str(initial10)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial10)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL38 0 INITIAL38 PWL(0 0 20P ' +str(initial1)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial1)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL39 0 INITIAL39 PWL(0 0 20P ' +str(initial1)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial1)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL40 0 INITIAL40 PWL(0 0 20P ' +str(initial5)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial5)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL41 0 INITIAL41 PWL(0 0 20P ' +str(initial12)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial12)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL42 0 INITIAL42 PWL(0 0 20P ' +str(initial3)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial13)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL43 0 INITIAL43 PWL(0 0 20P ' +str(initial11)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial11)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL44 0 INITIAL44 PWL(0 0 20P ' +str(initial2)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial2)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIAL45 0 INITIAL45 PWL(0 0 20P ' +str(initial6)+ ' '+ str(ENDINITIALTIME)+'p '+str(initial6)+' '+str(ENDINITIALTIME)+'p 0)\n')

    f.write('Iconst1 0 CONSTIN1 PWL(0 0 20P ' + (CONSTSET1) +' )\n')
    #f.write('Iconst2 0 CONSTIN2 PWL(0 0 20P +' +CONSTSET2+' )\n')
    #f.write('Iconst3 0 CONSTIN3 PWL(0 0 20P +' +CONSTSET+' )')
    #f.write('Iconst4 0 CONSTIN4 PWL(0 0 20P +' +CONSTSET+' )')
    #f.write('Iconst5 0 CONSTIN5 PWL(0 0 20P +' +CONSTSET+' )')


    f.write('IINITIALB1 0 INITIALB1 PWL(0 0 20P ' +str(initialb1)+ '*-22.6u'+' '+ str(ENDINITIALTIME)+'p '+str(initialb1)+ '*-22.6u'+' '+str(ENDINITIALTIME)+'p 0)\n')
    #f.write('IINITIALB2 0 INITIALB2 PWL(0 0 20P ' +str(initialb2)+ ' '+ str(ENDINITIALTIME)+'p '+str(initialb2)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIALB3 0 INITIALB3 PWL(0 0 20P ' +str(initialb1)+ ' '+ str(ENDINITIALTIME)+'p '+str(initialb1)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIALB4 0 INITIALB4 PWL(0 0 20P ' +str(initialb2)+ ' '+ str(ENDINITIALTIME)+'p '+str(initialb2)+' '+str(ENDINITIALTIME)+'p 0)\n')
    # f.write('IINITIALB5 0 INITIALB5 PWL(0 0 20P ' +str(initialb2)+ ' '+ str(ENDINITIALTIME)+'p '+str(initialb1)+' '+str(ENDINITIALTIME)+'p 0)\n')


    f.write('***   NEURONS ***\n')
    #f.write('X1 NEURONLETTER IN1 IN2 IN3 IN4 IN5 IN6 IN7 IN8 IN9 INB1          TARGETA INITIAL1 INITIAL2 INITIAL3 INITIAL4 INITIAL5 INITIAL6 INITIAL7 INITIAL8 INITIAL9 INITIALB1           A3 0 B3 0 DC3 0 DCc3 0 DOUTCOMPA CONSTIN1\n')
    f.write('X1 NEURONLETTER IN1 IN2 IN3 IN4 IN5 IN6 IN7 IN8 IN9 INB1          TARGETA INITIAL1 INITIAL2 INITIAL3 INITIAL4 INITIAL5 INITIAL6 INITIAL7 INITIAL8 INITIAL9 INITIALB1           A3 0 B3 0 DC3 0 DCc3 0 CONSTIN1\n')
    #f.write('X2 NEURONLETTER IN10 IN11 IN12 IN13 IN14 IN15 IN16 IN17 IN18 INB2 TARGETU INITIAL10 INITIAL11 INITIAL12 INITIAL13 INITIAL14 INITIAL15 INITIAL16 INITIAL17 INITIAL18 INITIALB2  A4 0 B4 0 DC4 0 DCC4 0 DOUTCOMPU CONSTIN2\n')
    #f.write('X3 NEURONLETTER IN19 IN20 IN21 IN22 IN23 IN24 IN25 IN26 IN27 INB3 TARGETI INITIAL19 INITIAL20 INITIAL21 INITIAL22 INITIAL23 INITIAL24 INITIAL25 INITIAL26 INITIAL27 INITIALB3  A5 A6 B5 B6 DC5 DC6 DCc5 DCC6 DOUTCOMPI CONSTIN3\n')
    #f.write('X4 NEURONLETTER IN28 IN29 IN30 IN31 IN32 IN33 IN34 IN35 IN36 INB4 TARGETE INITIAL28 INITIAL29 INITIAL30 INITIAL31 INITIAL32 INITIAL33 INITIAL34 INITIAL35 INITIAL36 INITIALB4  A6 A7 B6 B7 DC6 DC7 DCc6 DCC7 DOUTCOMPE CONSTIN4\n')
    #f.write('X5 NEURONLETTER IN37 IN38 IN39 IN40 IN41 IN42 IN43 IN44 IN45 INB5 TARGETO INITIAL37 INITIAL38 INITIAL39 INITIAL40 INITIAL41 INITIAL42 INITIAL43 INITIAL44 INITIAL45 INITIALB5  A7 0 B7 0 DC7 0 DCc7 0 DOUTCOMPO CONSTIN5\n')

    f.write('***   CONTROLS ***\n')
    f.write('.print devi LIN1\n')
    f.write('.print devi LIN2\n')
    f.write('.print devi LIN3\n')
    f.write('.print devi LIN4\n')
    f.write('.print devi LIN5\n')
    f.write('.print devi LIN6\n')
    f.write('.print devi LIN7\n')
    f.write('.print devi LIN8\n')
    f.write('.print devi LIN9\n')
    # f.write('.print devi LIN10\n')
    # f.write('.print devi LIN11\n')
    # f.write('.print devi LIN12\n')
    # f.write('.print devi LIN13\n')
    # f.write('.print devi LIN14\n')
    # f.write('.print devi LIN15\n')
    # f.write('.print devi LIN16\n')
    # f.write('.print devi LIN17\n')
    # f.write('.print devi LIN18\n')
    #f.write('.print devi LINBIAS1\n')
    #f.write('.print devi LINBIAS2\n')

    #f.write('.print devii Rin38\n')
    #f.write('.print devii Rin39\n')
    #f.write('.print devii Rin40\n')
    #f.write('.print devii Rin41\n')
    #f.write('.print devii Rin42\n')
    #f.write('.print devii Rin43\n')
    #f.write('.print devii Rin44\n')
    #f.write('.print devii Rin45\n')
    #f.write('.print devii Rin37\n')
    #f.write('.print devii Rin38\n')
    #f.write('.print devii Rin39\n')
    #f.write('.print devii Rin40\n')
    #f.write('.print devii Rin41\n')
    #f.write('.print devii Rin42\n')
    #f.write('.print devii Rin43\n')
    #f.write('.print devii Rin44\n')
    #f.write('.print devii Rin45\n')
    #f.write('.print devi rTARGETA\n')
    #f.write('.print devii iinbias\n')
    #f.write('.print phase LQ.XCOMP.X1\n')
    #f.write('.print phase LQ.xsyn1.x1\n')
    f.write('.print phase L1_q.xincr.xperceptron12.x1 \n')



    #f.write('.print phase L1_q.xdecr.xperceptron12.x1 \n')
    f.write('.print phase L2_q.xincr.xperceptron12.x1 \n')
    #f.write('.print phase L2_q.xdecr.xperceptron12.x1 \n')
    f.write('.print phase L4_q.xincr.xperceptron12.x1 \n')
    #f.write('.print phase L4_q.xdecr.xperceptron12.x1 \n')
    f.write('.print phase L5_q.xincr.xperceptron12.x1 \n')
    #f.write('.print phase L5_q.xdecr.xperceptron12.x1 \n')
 #   f.write('.print phase LQ.XTARGET.xperceptron14.x1 \n')
    #f.write('.print phase L1_q.xdecr.xperceptron14.x1 \n')
    #f.write('.print phase L2_q.xdecr.xperceptron14.x1 \n')
    #f.write('.print phase L4_q.xdecr.xperceptron14.x1 \n')
   # f.write('.print phase L5_q.xdecr.xperceptron14.x1 \n')

#    f.write('.print phase LOUT5.XCOMP.X1\n')
#    f.write('.print phase LOUT11.XCOMP.X1\n')
#    f.write('.print phase LQ.XTARGET.xperceptron15.x1 \n')
    #f.write('.print phase L1_q.xdecr.xperceptron15.x1 \n')
    #f.write('.print phase L2_q.xdecr.xperceptron15.x1 \n')
    #f.write('.print phase L4_q.xdecr.xperceptron15.x1 \n')
    #f.write('.print phase L5_q.xdecr.xperceptron15.x1 \n')
    #f.write('.print phase b1.xconvp.xconv1.x1\n')
    #f.write('.print phase b1.xconvm.xconv1.x1\n')
    #f.write('.print phase b2.xconvp.xconv1.x1\n')
    #f.write('.print phase b2.xconvm.xconv1.x1\n')
    #f.write('.print nodev wplus1.x1\n')
    #f.write('.print nodev wminus1.x1\n')
    f.write('.print devi ITARGETA\n')
    f.write('.print phase LQ.XCOMP.X1\n')
    f.write('.print devi ITARGETU\n')
    #f.write('.print phase LQ.XCOMP.X2\n')
    #f.write('.print devi RTARGETI\n')
    #f.write('.print phase LQ.XCOMP.X3\n')
    #f.write('.print devI RTARGETE\n')
    #f.write('.print phase LQ.XCOMP.X4\n')
    #f.write('.print devI RTARGETO\n')
    #f.write('.print phase LQ.XCOMP.X5\n')
    f.write('.print phase LSTORE1.XSYN1.X1\n')
    f.write('.print phase LSTORE1.XSYN2.X1\n')
    f.write('.print phase LSTORE1.XSYN3.X1\n')
    f.write('.print phase LSTORE1.XSYN4.X1\n')
    #f.write('.print phase LSTORE1.XSYN5.X1\n')
    #f.write('.print phase LSTORE1.XSYN6.X1\n')
    #f.write('.print phase LSTORE1.XSYN7.X1\n')
    #f.write('.print phase LSTORE1.XSYN8.X1\n')
    #f.write('.print phase LSTORE1.XSYN9.X1\n')
    #f.write('.print phase LSTORE1.XSYNBIAS.X1\n')

    # f.write('.print phase LQ.XBFRT1.XADJUST1.X1\n')
    # f.write('.print phase LQ.XBFRT2.XADJUST1.X1\n')
    # f.write('.print phase LQ.XBFRT3.XADJUST1.X1\n')
    # f.write('.print phase LQ.XBFRI3.XADJUST1.X1\n')
    # f.write('.print phase LQ.XBUFFACT2.XADJUST1.X1\n')
    #f.write('.print phase LQ.XCOMP.X5\n')
    # f.write('.print phase LQ.XSYN1.X1\n')
    # f.write('.print phase LQ.XSYN2.X1\n')
    # f.write('.print phase LQ.XSYN3.X1\n')
    # f.write('.print phase LQ.XSYN4.X1\n')
   # f.write('.print phase LQ.XSYN5.X1\n')
    # f.write('.print phase LQ.XSYN6.X1\n')
   # f.write('.print phase LQ.XSYN7.X1\n')
   # f.write('.print phase LQ.XSYN8.X1\n')
   # f.write('.print phase LQ.XSYN9.X1\n')
    # f.write('.print phase LQ.XSYNBIAS.X1\n')
    #f.write('.print phase LQ.XCOMP.X1\n')
    #f.write('.print phase LQ.XSYN1.X1\n')
    #f.write('.print phase LQ.XSYN2.X2\n')
    #f.write('.print phase LQ.XSYN3.X3\n')
    #f.write('.print phase LSTORE1.XSYN1.X1\n')
    # f.write('.print phase LSTORE1.XSYN1.X2\n')
    # f.write('.print phase LSTORE1.XSYN2.X2\n')
    # f.write('.print phase LSTORE1.XSYN3.X2\n')
    # f.write('.print phase LSTORE1.XSYN4.X2\n')
    # f.write('.print phase LSTORE1.XSYN5.X2\n')
    # f.write('.print phase LSTORE1.XSYNBIAS.X2\n')
    #f.write('.print phase LSTORE1.XSYN1.X3\n')
    #f.write('.print phase LQ.XCONVP.XCONVB.X1\n')
    #f.write('.print phase Lstore1.XSYN1.X1\n')
    #f.write('.print phase Lstore1.XSYN2.X1\n')
    #f.write('.print devv VDCConv\n')
    #f.write('.print phase Lstore1.XSYN1.X1\n')
    #f.write('.print phase Lstore1.XSYNBIAS.X1\n')

print(str(endtime)+'s')
start_time = time.time()
os.system('josim-cli -o genneuron.csv genneuron.cir -V 1')
print("--- %s seconds ---" % (time.time() - start_time))
os.system('python3 josim-plot genneuron.csv -t stacked')
