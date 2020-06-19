import pandas as pd
import numpy as np
import math

from pandas import DataFrame
from scipy.signal import butter, filtfilt #Import the extra module required
#Define the filter
def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = filtfilt(b, a, data)
    return y


ppg0101 = pd.read_csv("C:\\Downloads\\gamer1-ppg-2000-01-01.csv")
ppg0101['Time']='2000-01-01 '+ppg0101['Time']
ppg0102 = pd.read_csv("C:\\Downloads\\gamer1-ppg-2000-01-02.csv")
ppg0102['Time']='2000-01-02 '+ppg0102['Time']

bpmTotal=[]
pnn20Total=[]
pnn50Total=[]
lfTotal=[]
hfTotal=[]
ratio=[]
time=[]
for day in range(1,3):
    if day==1:
        ppgTotal = ppg0101
    else:
        ppgTotal=ppgTotal.append(ppg0102)
    for hour in range(0, 24):
        for minute in range(0,60):
            if (day==1 and hour>12) or(day==2 and hour<11):
                if hour<10:
                    if minute<10:
                        temp='2000-01-0'+day.__str__()+' 0'+hour.__str__()+':0'+minute.__str__()+':00'
                    else:
                        temp = '2000-01-0' + day.__str__() + ' 0' + hour.__str__() + ':'+ minute.__str__()+':00'
                else:
                    if minute < 10:
                        temp='2000-01-0'+day.__str__()+' '+hour.__str__()+':0'+minute.__str__()+':00'
                    else:
                        temp = '2000-01-0' + day.__str__() + ' ' + hour.__str__() + ':'+ minute.__str__()+':00'
                print(temp)
                ppg= ppgTotal.loc[ppgTotal['Time']<temp,:]
                ppgTotal=ppgTotal.drop(ppgTotal.loc[ppgTotal['Time']<temp].index)
                if not ppg.empty:
                    ppg['Time']=pd.to_datetime(ppg['Time'])
                    time.append(temp)
                    filtered = butter_lowpass_filter(ppg.Red_Signal, 2, 50.0, 5)#Butterworth filter
                    ppg['filtered']=filtered
                    hrw = 0.75
                    fs = 100
                    moving_average = ppg['filtered'].rolling(int(hrw*fs)).mean()
                    average = (np.mean(ppg.filtered))
                    moving_average = [average if math.isnan(x) else x for x in moving_average]#take total average if moving average is null
                    window = []
                    peaklist = []
                    pos = 0 #position
                    for datapoint in ppg.filtered:
                        if pos<ppg['Time'].size:
                            if (datapoint <= moving_average[pos]) and (len(window) < 1): #from start when data is below moving average, no need to add into window
                                pos += 1
                            elif (datapoint > moving_average[pos]): #when signal is above, add it into window, from which we get the maximum as peak
                                window.append(datapoint)
                                pos += 1
                            else: #when signal is below moving average, we get peak from the current window
                                maximum = max(window)
                                beatposition = pos - len(window) + (window.index(max(window))) #calculate the time for max in the window
                                peaklist.append(beatposition)
                                window = [] #start over
                                pos += 1
                    PP_list = []
                    count = 0
                    while (count < (len(peaklist)-1)):
                        PP_interval = (peaklist[count+1] - peaklist[count]) #Calculate distance between peaks
                        ms_dist = ((PP_interval / fs) * 1000.0) #convert unit into millisecond
                        PP_list.append(ms_dist)
                        count += 1
                    #outlier, set 500 ms as thresholds
                    upper_threshold = (np.mean(PP_list) + 500)
                    lower_threshold = (np.mean(PP_list) - 500)
                    count = 0
                    peaklist_cor = []
                    PP_list_cor = []
                    while count < len(PP_list):
                        if (PP_list[count] < upper_threshold) and (PP_list[count] > lower_threshold):
                            PP_list_cor.append(PP_list[count])
                            peaklist_cor.append(peaklist[count])
                        count += 1
                    if np.mean(PP_list_cor)>0:
                        bpm = 60000 / np.mean(PP_list_cor) #60000 ms (1 minute) / average PP interval of signal
                    else:
                        bpm=0
                    bpmTotal.append(bpm)
                    PP_diff = []
                    PP_sqdiff = []
                    count = 1
                    while (count < (len(PP_list_cor)-1)):
                        PP_diff.append(abs(PP_list_cor[count] - PP_list_cor[count+1]))
                        count += 1
                    nn20 = [x for x in PP_diff if (x>20)]
                    nn50 = [x for x in PP_diff if (x>50)]
                    if len(PP_diff)>0:
                        pnn20 = float(len(nn20)) / float(len(PP_diff))
                    else:
                        pnn20=0
                    if len(PP_diff)>0:
                        pnn50 = float(len(nn50)) / float(len(PP_diff))
                    else:
                        pnn50=0
                    pnn20Total.append(pnn20)
                    pnn50Total.append(pnn50)
                    PP_x = peaklist_cor
                    if len(PP_x)>3:
                        from scipy.interpolate import interp1d
                        n = len(ppg.Red_Signal)
                        PP_y = PP_list_cor
                        PP_x_new = np.linspace(PP_x[0],PP_x[-1],n) #Create evenly spaced timeline starting at the second peak
                        f = interp1d(PP_x, PP_y, kind='cubic') #Interpolate the signal w
                        frq = np.fft.fftfreq(n, d=((1/100)))
                        frq = frq[range(int(n/2))]
                        Y = np.fft.fft(f(PP_x_new))/n #Calculate FFT
                        Y = Y[range(int(n/2))] #Return one side of the FFT
                        lf = np.trapz(abs(Y[(frq>=0.04) & (frq<=0.15)])) #LF:0.04 and 0.15Hz (LF)
                        hf = np.trapz(abs(Y[(frq>=0.16) & (frq<=0.5)])) #HF:0.16-0.5Hz (HF)
                        ratio.append(lf/hf)
                    else:
                        lf=0
                        hf=0
                        ratio.append(0)
                    lfTotal.append(lf)
                    hfTotal.append(hf)
data = {'Time':time,
        'bpm':bpmTotal,
       'pnn20':pnn20Total,
       'pnn50':pnn50Total,
        'lf':lfTotal,
        'hf':hfTotal,
        'ratio':ratio}
df = DataFrame(data)
df.to_csv("C:\\Downloads\\gamer1.csv")
