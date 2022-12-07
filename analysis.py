import numpy as np
import cv2

visual_input_append=np.load("visual_input_append.npy")
spike_data=np.load("spike_data.npy")

framenum=visual_input_append.shape[0]
calcunum=spike_data.shape[0]
RF_TIME_LENGTH=10

spike_count=0
# スパイクが発火したタイミングを捜索し、100ms前の値を取得する。
for i in range(calcunum):
    if spike_data[i,4]==1:
        # スパイクが発火した回数
        spike_count=spike_count+1
        j=int(i/10)
        if j>RF_TIME_LENGTH:
            if spike_count==1:            
                visual_input_affecting_spike=[visual_input_append[j-10:j,:,:]]
            else:
                visual_input_affecting_spike=np.append(visual_input_affecting_spike, [visual_input_append[j-10:j,:,:]],axis=0) 


                
# STA分析→平均を求めている
sta_result=np.average(visual_input_affecting_spike, axis=0)
print(spike_count)

for i in range(10):
    staforviewer=cv2.resize(sta_result[i,:,:]*2,[300,300])
    cv2.imshow('sta',staforviewer)



    #繰り返し分から抜けるためのif文
    key =cv2.waitKey(1000)
    if key == 27:
        break