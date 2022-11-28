from os import access
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
# 1次のIIRフィルタによる重みづけ
alpha_h=0.95312500
# ２相性の時間フィルタによる重みづけと係数
alpha_a1=0.875
alpha_a2=2*alpha_a1-1
alpha_a3=3*alpha_a1-2
k_b1=-1      
k_b2=2
k_b3=-1
# 単相性フィルタによる重みづけ
alpha_ac1=0.8
alpha_ac2=2*alpha_ac1-1
k_a1=2      
k_a2=-1
# 黒のサイズ
W=int(100)
H=int(100)
blank = np.zeros((W, H),np.float32)
# 白のサイズ
W1=int(20)
H1=int(20)
# CSVファイルに格納する配列
s=np.zeros((45,4), np.float32)
# 水平細胞の初期値
horizontal_cell_old=0
# 双極細胞の初期値
bipolar_cell1=0
bipolar_cell_old1=0
bipolar_cell_old2=0
bipolar_cell_old3=0
# ON型の双極細胞
b_on_old1=0
b_on_old2=0
b_on=0
# アマクリン細胞の初期値
AC2=0   
a=0
b1=0
b2=0
b3=0
count=0
timing1=1
timing2=6
# スパイクパラメータ
dt = 0.5; T = 200 # ms
nt = round(T/dt) # シミュレーションステップ数
t1 = np.arange(nt)*dt
v_arr = np.zeros(nt) # 膜電位を記録する配列
u_arr = np.zeros(nt) # 回復変数を記録する配列
count1=np.zeros(nt)
C = 23 # 膜容量 (pF)
a_spike = 0.02 # 回復時定数の逆数 (1/ms)
b_spike = 0.2 # u の v に対する共鳴度合い (pA/mV)
c_spike=-65
d_spike = 8 # 発火で活性化される正味の外向き電流 (pA)
vrest = 50 # 静止膜電位 (mV)
vreset = 60 # リセット電位 (mV)
vthr = -43 # 閾値電位 (mV)
vpeak = 30 #　ピーク電位 (mV)
g_arr = np.zeros(nt)
v_old=-65.75
u_old=7.76
v = 0; v_ = v; u = 0
Cm=23
hk=np.zeros((W, H),np.float32)

for i in range(0, nt):
    
    #インパルス応答変化一定間隔応答
    # if 1<=count<=100:
    # if count==300:
    if 0<=count<=50:
    # if count>=500:
    # # if i%50==0:
        blank[40:60,40:60] = np.ones((W1,H1),np.float32)
        a1=blank
        a=a1
        g2=15.875
        # g2=7
    elif 100<=count<=150:
        blank[40:60,40:60] = np.ones((W1,H1),np.float32)
        a1=blank
        a=a1
        g2=15.875
    elif 200<=count<=250:
        blank[40:60,40:60] = np.ones((W1,H1),np.float32)
        a1=blank
        a=a1
        g2=15.875
    elif 300<=count<=350:
        blank[40:60,40:60] = np.ones((W1,H1),np.float32)
        a1=blank
        a=a1
        g2=15.875
    # elif 900<=count<=1000:
    #     blank[40:60,40:60] = np.ones((W1,H1),np.float32)
    #     a1=blank
    #     a=a1
    #     g2=-15.875
    else:
        a=blank
        g2=-15.875
        # g2=0
    a=blank
    #視細胞
    photoreceptor_cell = cv2.GaussianBlur(a, (7, 7), 1.0)
    #水平細胞
    horizontal_cell= cv2.GaussianBlur(photoreceptor_cell,(19, 19), 3.0)
    horizontal = horizontal_cell_old*alpha_h+horizontal_cell*(1-alpha_h)
    horizontal_cell_old =  horizontal 
    #双極細胞
    bipolar_cell = photoreceptor_cell-horizontal
    b11=bipolar_cell_old1*alpha_a1+bipolar_cell*(1-alpha_a1)
    b12=bipolar_cell_old2*alpha_a2+bipolar_cell*(1-alpha_a2)
    b13=bipolar_cell_old3*alpha_a3+bipolar_cell*(1-alpha_a3)
    sum_b=k_b1*b11+k_b2*b12+ k_b3*b13
    bipolar_cell_old1=b11
    bipolar_cell_old2=b12
    bipolar_cell_old3=b13
    #ギャップ結合した双極細胞
    sum_b1 = cv2.GaussianBlur(sum_b, (7, 7), 1.0)
    sum_b2=sum_b1-AC2
    # 半波整流
    hk=sum_b2>0
    hk1=hk*sum_b2
    b_on=hk1
    g=b_on[50,50]
    a2=a[50,50]
    # アマクリン細胞
    AC=b_on_old1*alpha_ac1+b_on*(1-alpha_ac1)
    AC1=b_on_old2*alpha_ac2+b_on*(1-alpha_ac2)
    # AC2=k_b2*AC+k_b3*AC1
    AC2=k_a1*AC+k_a2*AC1
    b_on_old1=AC
    b_on_old2=AC1
    
    #スパイク発火に向けて
    g1=g*255
    # g2=0.0392*g1#i=10
    g1=0.0588*g1
    #i=15
    # g2=32*g1
    # if g2>=511.99:
    #     g2=511.99
    # g2=2.625
    # g2=7
    # スパイク発火オイラー
    v=v_old+(0.04*v_old*v_old+5*v_old+140-u_old+g2)*(dt)
    # v=v_old+(0.04*v_old*v_old+5*v_old+140-u_old+g2)*(dt/Cm)
    u=u_old+a_spike*(b_spike*v_old-u_old)*dt

    # v=v_old+(dt/Cm)*((v_old+51.8)*(v_old+43)+(g2-u_old))
    # u=u_old+dt*((-0.043*0.4)*(v_old+51.8)-u_old)

    if v>=30:
        # v=-52
        v=c_spike
        # u=u-56
        u=u+d_spike
    v_old=v
    u_old=u
    v_arr[i] = v # 膜電位の値を保存
    u_arr[i] = u # 回復変数の値を保存
    g_arr[i] = g2
    count +=1
    blank = np.zeros((W, H),np.float32)
    g2=0
t1 = np.arange(nt)*dt
np.savetxt('a.csv',v_arr , delimiter=",",fmt='%12.6f')
np.savetxt('b.csv',u_arr , delimiter=",",fmt='%12.6f')
np.savetxt('c.csv',g_arr , delimiter=",",fmt='%12.6f')
# np.savetxt('b.csv',count1 , delimiter=",",fmt='%12.6f')
# fig,ax1=plt.subplots(figsize=(20, 20))
# # plt.figure(figsize=(50, 50))
# ax1.plot(t1,v_arr ,"k")
# ax1.set_ylabel("v (mV)")
# ax2 = ax1.twinx()
# # ax2.plot(t1,g_arr  ,"k")
# ax2.set_ylabel("Input")
# 複数表示
plt.subplot(3,1,1)
plt.plot(t1, v_arr, color="k")
plt.ylabel('v (mV)')
plt.subplot(3,1,2)
plt.plot(t1, u_arr, color="k")
plt.xlabel('Time (ms)')
plt.ylabel('u(pA)')
plt.subplot(3,1,3)
plt.plot(t1, g_arr, color="k")
plt.xlabel('Time (ms)')
plt.ylabel('Input')
plt.show()
