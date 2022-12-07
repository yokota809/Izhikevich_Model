from os import access
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import keyboard
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
W1=int(1)
H1=int(1)
# 水平細胞の初期値
horizontal_cell_old=np.ones((W, H),np.float32)*0.5
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
# 細胞応答
# fig = plt.figure(figsize=(9,6))
# plt.get_current_fig_manager().window.wm_geometry("+1000+400")
# plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, hspace=0.2)
# 余白を設定
# plt.yticks(np.arange(-1.0,1.0, step=0.2))

#時間フィルタ
# plt.subplots_adjust(wspace=0.2, hspace=0.4)
# # plt.title("Temporal Filter")
# ax11 = fig.add_subplot(221, xlim=[0, 40], ylim=[-1.5, 1.5], xlabel="frame number", ylabel="intensity[a.u.]")
# # ax11 = fig.add_subplot(221, xlim=[0, 30], ylim=[-1.5, 1.5], xlabel="t(s)", ylabel="intensity[a.u.]")
# plt.title("PC")
# # plt.title("photoreceptor")
# ax12 = fig.add_subplot(222, sharex=ax11, ylim=[0, 0.05], xlabel="frame number", ylabel="intensity[a.u.]")
# plt.title("HC")
# # plt.title("horizontal")
# ax13 = fig.add_subplot(223, xlim=[0, 40], ylim=[-0.05, 0.05], xlabel="frame number", ylabel="intensity[a.u.]")
# # ax13 = fig.add_subplot(223, xlim=[0, 30], ylim=[-1.0, 1.0], xlabel="t(s)", ylabel="intensity[a.u.]")
# plt.title("BC")
# # plt.title("bipolar")
# ax14 = fig.add_subplot(224, sharex=ax11, ylim=[-0.05, 0.05], xlabel="frame number", ylabel="intensity[a.u.]")
# # ax14 = fig.add_subplot(224, sharex=ax11, ylim=[-1.0, 1.0], xlabel="t(s)", ylabel="intensity[a.u.]")
# plt.title("AC")
# # plt.title("amacrine")

# 空間フィルターのグラフ設定
# fig1 = plt.figure(figsize=(9,6))
# plt.get_current_fig_manager().window.wm_geometry("+0+400")
# plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, hspace=0.2)
# plt.subplots_adjust(wspace=0.2, hspace=0.4)
# # plt.title("Spatial Filter")
# ax15 = fig1.add_subplot(221, xlim=[0, 100], ylim=[-0.5, 1.5], xlabel="frame number", ylabel="intensity[a.u.]")
# plt.title("PC")
# # plt.title("photoreceptor")
# ax6 = fig1.add_subplot(222, sharex=ax15, ylim=[-0.02, 0.04], xlabel="frame number", ylabel="intensity[a.u.]")
# plt.title("HC")
# # plt.title("horizontal")
# # ax7 = fig1.add_subplot(223, xlim=[0, 128], ylim=[-2, 2], xlabel="frame number", ylabel="intensity[a.u.]")
# ax7 = fig1.add_subplot(223, xlim=[0, 100], ylim=[-0.02, 0.02], xlabel="frame number", ylabel="intensity[a.u.]")
# ax7 = fig1.add_subplot(223, sharex=ax15, sharey=ax15, xlabel="t(s)", ylabel="intensity[a.u.]")
# plt.title("BC")
# # plt.title("bipolar")
# ax8 = fig1.add_subplot(224, sharex=ax15, ylim=[-0.02, 0.02], xlabel="frame number", ylabel="intensity[a.u.]")
# plt.title("AC")
# # plt.title("amacrine")
start = time.time()
jiku=range(1, 101)

np.random.seed(seed=0)

# スパイクパラメータ
dt = 0.5; T = 200*10 # ms
nt = round(T/dt) # シミュレーションステップ数
nt1 = round(T/dt*10)

count1=np.zeros(nt)
t1 = np.arange(nt)*dt
count1=np.zeros(nt)
a_spike = 0.02 # 回復時定数の逆数 (1/ms)
b_spike = 0.2 # u の v に対する共鳴度合い (pA/mV)
c_spike=-60
d_spike = 8 # 発火で活性化される正味の外向き電流 (pA)
vrest = 50 # 静止膜電位 (mV)
vreset = 60 # リセット電位 (mV)
vthr = -43 # 閾値電位 (mV)
vpeak = 30 #　ピーク電位 (mV)
t_spike = np.arange(nt)*dt
v_old=-65.75
u_old=7.76
# # 初期化 (膜電位, 膜電位 (t-1), 回復電流)
v = 60; v_ = v; u = 0
v_arr = np.zeros(nt1,np.float32) # 膜電位を記録する配列
u_arr = np.zeros(nt1,np.float32) # 回復変数を記録する配列
RF_sum = np.zeros(nt1,np.float32)
# CSVファイルに格納する配列
s=np.zeros((nt,5), np.float32)
spike_data=np.zeros((nt1,5), np.float32)
sta_data=np.zeros((W,H,nt), np.float32)
# s=np.zeros((50,5), np.float32)
# 画面表示
# fig2=plt.figure(figsize=(12,3))
# plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=1)
# plt.subplots_adjust(wspace=0.2, hspace=0.4)
# plt.get_current_fig_manager().window.wm_geometry("+10+10")
# plt.title("Model")
x=1
y=5
# スパイク表示
# fig10=plt.figure()
# ax15 = fig10.add_subplot(2,1,1)
# ax16 = fig10.add_subplot(2,1,2, ylim=[-0.2, 0.1], xlabel="t(s)")
# plt.plot(t_spike, v_arr, color="k")
# plt.ylabel('Membrane potential (mV)')
# plt.subplot(2,1,2)
# plt.plot(t_spike, u_arr, color="k")
# plt.xlabel('Time (ms)')
# plt.ylabel('Recovery current (pA)')
hk=np.zeros((W, H),np.float32)
# スパイクニューロンモデルの入力のスケールパラメータ
scale_x=32
scale_gain=4
# 視細胞とギャップ結合の双極細胞の空間スケールパラメータ
# photoreceptor_kernel = np.ones((5,5),np.float32)/25
# horizontal_kernel = np.ones((19, 19),np.float32)/361
g_arr = np.zeros(nt1)

lnstart = time.time()
for i in range(0, nt):
    # if 200<=i<=400:
    visual_input=np.random.uniform(0, 1, (H,W))
    if i==0:
        visual_input_append=[visual_input]
    else:
        visual_input_append = np.append(visual_input_append, [visual_input],axis=0) 

    #インパルス応答変化一定間隔応答
    # if i%50==0:
    # if count==300:
    # if 0<=i<=100:
    #     random_pattern=np.ones((W, H),np.float32)
    # elif 300<=i<=400:
    #     random_pattern=np.ones((W, H),np.float32)
    # # if count==0:
    #     # blank = np.ones((W,H),np.float32)
    #     blank[50,50] = np.ones((W1,H1),np.float32)
    #     # blank[40:60,40:60] = np.ones((20,20),np.float32)
    #     a1=blank*10
    #     a=a1
        # print('on')
    # # if count<timing2:
    # #     a1=blank
    # #     blank[65:85,65:85] = np.ones((W1,H1),np.float32)
    # #     a=a1
    # else:
    #     a=np.zeros((W, H),np.float32)
    # if count<timing1:
    #     a1=blank
    #     blank[40:60,40:60] = np.ones((W1,H1),np.float32)
    #     a=a1
    # if count<timing2:
    #     a1=blank
    #     blank[65:85,65:85] = np.ones((W1,H1),np.float32)
    #     a=a1
    # else:
    #     # a=np.zeros((W, H),np.float32)
    #     random_pattern=np.zeros((W, H),np.float32)
        # print('off')
    # print(a[50,50])   
    #視細胞
        #a2 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    
    # photoreceptor_cell = cv2.filter2D(random_pattern,-1,photoreceptor_kernel)
    photoreceptor_cell = cv2.GaussianBlur(visual_input, (5, 5), 1.0)
    # photoreceptor_cell = cv2.filter2D(a,-1,photoreceptor_kernel)
    cs_p=photoreceptor_cell[50,:]
    # np.savetxt('ph1IIR_filter1.csv',cs_p ,delimiter=",")
    # cv2.imshow('a',photoreceptor_cell)
    # photoreceptor_cell = cv2.GaussianBlur(a, (7, 7), 1.0)
    #水平細胞
    # horizontal_cell = cv2.filter2D(photoreceptor_cell,-1,horizontal_kernel)
    horizontal_cell = cv2.GaussianBlur(photoreceptor_cell,(19, 19), 3.0)
    cs_h=horizontal_cell[50,:]
    # np.savetxt('h1IIR_filter1.csv',cs_h ,delimiter=",")
    # cv2.imshow('a',horizontal_cell)
    # horizontal_cell= cv2.GaussianBlur(photoreceptor_cell,(19, 19), 3.0)
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
    #ギャップ結合した双極細胞とアマクリン細胞の抑制
    # sum_b1 = cv2.filter2D(sum_b,-1,photoreceptor_kernel)
    sum_b1 = cv2.GaussianBlur(sum_b, (7, 7), 1.0)
    sum_b2=sum_b1-AC2
    cs_b=sum_b2[50,:]
    # np.savetxt('b1IIR_filter1.csv',sum_b2 ,delimiter=",")
    # cv2.imshow('a',sum_b2)
    # np.savetxt('Test.csv',sum_b2,fmt='%12.3f' )
    # 半波整流
    hk=sum_b2>0
    hk1=hk*sum_b2
    b_on=hk1
    sum_b_on = cv2.boxFilter(b_on,ddepth=-1,ksize=(7,7),normalize=False)
    input_g=sum_b_on[50,50]
    # print(input_g)
    AC=b_on_old1*alpha_ac1+b_on*(1-alpha_ac1)
    AC1=b_on_old2*alpha_ac2+b_on*(1-alpha_ac2)
    # AC2=k_b2*AC+k_b3*AC1
    AC2=k_a1*AC+k_a2*AC1
    b_on_old1=AC
    b_on_old2=AC1
    SpikingNeuron_input=input_g*scale_gain*scale_x
    # print(SpikingNeuron_input)
    # スパイク生成処理
    for j  in range(0,10):

        v=v_old+(0.04*v_old*v_old+5*v_old+140-u_old+SpikingNeuron_input)*(dt)
        u=u_old+a_spike*(b_spike*v_old-u_old)*dt
        if v>=vpeak:
            v=c_spike
            u=u+d_spike
            RF=1
        else :
            RF=0  
        v_old=v
        u_old=u
        v_arr[i*10+j] = v # 膜電位の値を保存
        u_arr[i*10+j] = visual_input[50,50] # 回復変数の値を保存
        g_arr[i*10+j] = SpikingNeuron_input
        RF_sum[i*10+j] = RF
        spike_data[i*10+j][0]=visual_input[50,50]
        spike_data[i*10+j][1]=SpikingNeuron_input
        spike_data[i*10+j][2]=v
        spike_data[i*10+j][3]=u
        spike_data[i*10+j][4]=RF
    # spike_data[i][4]=SpikingNeuron_input
   
    
    #画像表示
    # ax1=fig2.add_subplot(x,y,1, xlim=[0, 100], ylim=[0, 100])
    # ax1.set_title('PC')
    # # ax1.set_title('Photoreceptor',fontsize=12)
    # plt.imshow(photoreceptor_cell,cmap="gray")
    # ax2=fig2.add_subplot(x,y,2, xlim=[0, 100], ylim=[0, 100])
    # ax2.set_title('HC')
    # plt.imshow(horizontal,cmap="gray")
    # ax3=fig2.add_subplot(x,y,3, xlim=[0, 100], ylim=[0, 100])
    # ax3.set_title('BC')
    # # plt.imshow(bipolar_cell,cmap="gray")
    # plt.imshow(sum_b,cmap="gray")
    # ax4=fig2.add_subplot(x,y,4, xlim=[0, 100], ylim=[0, 100])
    # ax4.set_title('Bon')
    # # plt.imshow(sum_b,cmap="gray")
    # plt.imshow(b_on,cmap="gray")
    # ax5=fig2.add_subplot(x,y,5, xlim=[0, 100], ylim=[0, 100])
    # ax5.set_title('AC')
    # plt.imshow(AC2,cmap="gray")
    # plt.imshow(sum_b12,cmap="gray")
    
    # 時間フィルター表示
    # #入力画像
    t_p=photoreceptor_cell[50,50]
    t_h=horizontal[50,50]
    t_bon=b_on[50,50]
    t_ac=AC2[50,50]
    # ax11.scatter(count, tp)
    # ax12.scatter(count, t_h)
    # ax13.scatter(count, t_b)
    # ax14.scatter(count, t_ac)
    s[i][0]=t_p
    s[i][1]=t_h
    s[i][2]=t_bon
    s[i][3]=t_ac
    s[i][4]=SpikingNeuron_input
    np.savetxt('IIR_filter.csv',s , delimiter=",",fmt='%12.6f')
    # plt.plot(tp)
    # plt.plot(t_h)
    # plt.plot(t_b)
    # plt.plot(t_ac)
    
    t_spike=count*dt
    #空間フィルタ
    cs_p=photoreceptor_cell[50,:]
    cs_h=horizontal[50,:]
    # cs_b1=bipolar_cell[64,:]
    cs_b1=b_on[50,:]
    # ギャップ結合しているやつ
    cs_bon1=sum_b2[50,:]
    cs_bon=AC2[50,:]
    np.savetxt('ph1IIR_filter1.csv',cs_p ,delimiter=",")
    np.savetxt('hIIR.csv',cs_h , delimiter=",",fmt='%12.6f')
    np.savetxt('bonIIR.csv',cs_b1 , delimiter=",",fmt='%12.6f')
    np.savetxt('sumb2IIR.csv',cs_bon1 , delimiter=",",fmt='%12.6f')
    np.savetxt('AC2IIR.csv',cs_bon , delimiter=",",fmt='%12.6f')
    
    if count==100:
    #     # fig.savefig("graph.png")
        np.savetxt('sample1.csv',cs_p , delimiter=",")
        np.savetxt('sample2.csv',cs_h ,delimiter=",")
        np.savetxt('sample3.csv',sum_b2[50,:] ,delimiter=",")
        # np.savetxt('IIR_filter4.csv',cs_bon , delimiter=",",fmt='%12.6f')
        # np.savetxt('IIR_filter5.csv',cs_bon1 , delimiter=",",fmt='%12.6f')
    
    # Line1=ax15.plot(jiku, cs_p,color="b")
    # Line2=ax6.plot(jiku, cs_h,color="b")
    # Line3=ax7.plot(jiku, cs_b1,color="b")
    # Line4=ax8.plot(jiku, cs_bon,color="b")
    plt.pause(0.01)     
    # line1 = Line1.pop(0)
    # line2 = Line2.pop(0)
    # line3 = Line3.pop(0)
    # line4 = Line4.pop(0)
    # line1.remove()
    # line2.remove()
    # line3.remove()
    # line4.remove()
    count +=1
    
    # print(t)
    # if count==1:
    #     # fig.savefig("graph.png")
    #     # np.savetxt('114IIR_filter1.csv',cs_h , delimiter=",",fmt='%12.6f')
    #     break
    # if count==1:
    #     # fig.savefig("graph.png")
    #     np.savetxt('ph1IIR_filter1.csv',cs_p , delimiter=",",fmt='%12.6f')
    #     np.savetxt('IIR_filter114.csv',cs_h , delimiter=",",fmt='%12.6f')
    #     np.savetxt('IIR_filter3.csv',cs_b1 , delimiter=",",fmt='%12.6f')
    #     # np.savetxt('IIR_filter4.csv',cs_bon , delimiter=",",fmt='%12.6f')
    #     # np.savetxt('IIR_filter5.csv',cs_bon1 , delimiter=",",fmt='%12.6f')
    #     break
    # if count==39:
    #     fig10.savefig("graph.png")
    #     np.savetxt('a.csv',v_arr , delimiter=",",fmt='%12.6f')
    #     np.savetxt('b.csv',u_arr , delimiter=",",fmt='%12.6f')
    # if count>40:
    #     fig10.savefig("graph.png")
    #     np.savetxt('sample.csv',v_arr , delimiter=",",fmt='%12.6f')
    #     np.savetxt('sample1.csv',u_arr , delimiter=",",fmt='%12.6f')
        # np.savetxt('IIR_filter1.csv',cs_p , delimiter=",",fmt='%12.6f')
        # np.savetxt('IIR_filter2.csv',cs_h , delimiter=",",fmt='%12.6f')
        # np.savetxt('IIR_filter3.csv',cs_b1 , delimiter=",",fmt='%12.6f')
        # np.savetxt('IIR_filter4.csv',cs_bon , delimiter=",",fmt='%12.6f')
        # np.savetxt('IIR_filter5.csv',cs_bon1 , delimiter=",",fmt='%12.6f')
    #     break
    # if cv2.waitKey(1) == 27:
    #     break
    # np.savez('data_3jigen1206',sta_data)
    # sta_data=np.zeros((W,H,nt), np.float32)
t = time.time() - start
print('t',t)
print("finish")
# t1 = np.arange(nt)*dt
    # ax14.scatter(count, t_ac)
np.save('visual_input_append1207.npy',visual_input_append)
np.save('spike_data1207.npy',spike_data)
# spike_data[i][0]=random_pattern[50,50]
# spike_data[i][1]=SpikingNeuron_input
# spike_data[i][2]=v
# spike_data[i][3]=u
# spike_data[i][4]=RF
# np.savez('data_sample1206',g_arr,v_arr,u_arr,RF_sum)
# np.savetxt('a.csv',v_arr , delimiter=",",fmt='%12.6f')
# t1= round(T/dt*10)
t1= np.arange(nt1)*dt
np.savetxt('Spike.csv',spike_data , delimiter=",",fmt='%12.6f')
plt.figure(figsize=(5, 5))
plt.subplot(3,1,1)
plt.plot(t1, g_arr, color="k")
plt.ylabel('v (mV)')
plt.subplot(3,1,2)
plt.plot(t1, v_arr, color="k")
plt.xlabel('Time (ms)')
plt.ylabel('u(pA)')
plt.subplot(3,1,3)
plt.plot(t1,RF_sum, color="k")
plt.xlabel('Time (ms)')
plt.ylabel('Input')
plt.show()
