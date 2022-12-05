import numpy as np
import matplotlib.pyplot as plt
count=0
# スパイクパラメータ
dt = 0.5; T = 400 # ms
nt = round(T/dt) # シミュレーションステップ数
t1 = np.arange(nt)*dt
v_arr = np.zeros(nt) # 膜電位を記録する配列
u_arr = np.zeros(nt) # 回復変数を記録する配列
count1=np.zeros(nt)
# パラメータ
a_spike = 0.02 # 回復時定数の逆数 (1/ms)
b_spike = 0.2 # u の v に対する共鳴度合い (pA/mV)
c_spike=-60
d_spike = 8 # 発火で活性化される正味の外向き電流 (pA)
vrest = 50 # 静止膜電位 (mV)
vreset = 60 # リセット電位 (mV)
vthr = -43 # 閾値電位 (mV)
vpeak = 30 #　ピーク電位 (mV)

g_arr = np.zeros(nt)
v_old=-60
u_old=0
# v_old=-65.75
# u_old=7.76
v = 0; u = 0
i_input=np.float64(60)
# i_input=15.875,np.float64
x = round(i_input, 1)
# print(x)
# I=0,np.float64
# print(type(i_input))
for i in range(0, nt):
    
    #インパルス応答変化一定間隔応答
    # if 1<=count<=100:
    # if count==300:
    if 0<=count<=50:
    # if count>=500:
    # # if i%50==0:
        # I=i_input
        I=0
        # g2=7
    elif 200<=count<=600:
        I=i_input
        
    # elif 200<=count<=250:
    #     I=i_input
    # elif 300<=count<=350:
    #     I=i_input
    # elif 900<=count<=1000:
    #     g2=-15.875
    else:
        # I=-1*i_input
        I=0
        # g2=0
    # スパイク発火オイラー
    v=v_old+(0.04*v_old*v_old+5*v_old+140-u_old+I)*(dt)
    u=u_old+a_spike*(b_spike*v_old-u_old)*dt
    if v>=vpeak:
        v=c_spike
        u=u+d_spike
    v_old=v
    u_old=u
    v_arr[i] = v # 膜電位の値を保存
    u_arr[i] = u # 回復変数の値を保存
    g_arr[i] = I
    count +=1
t1 = np.arange(nt)*dt
np.savetxt('a.csv',v_arr , delimiter=",",fmt='%12.6f')
np.savetxt('b.csv',u_arr , delimiter=",",fmt='%12.6f')
np.savetxt('c.csv',g_arr , delimiter=",",fmt='%12.6f')
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
