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
s=np.zeros((51,5), np.float32)
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
dt = 0.5; T = 500 # ms
nt = round(T/dt) # シミュレーションステップ数
count1=np.zeros(nt)
t1 = np.arange(nt)*dt
count1=np.zeros(nt)
a_spike = 0.02 # 回復時定数の逆数 (1/ms)
b_spike = 0.2 # u の v に対する共鳴度合い (pA/mV)
c_spike=-65
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
v_arr = np.zeros(nt,np.float32) # 膜電位を記録する配列
u_arr = np.zeros(nt,np.float32) # 回復変数を記録する配列
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

for i in range(0, 50):
        
    #インパルス応答変化一定間隔応答
    # if i%50==0:
    # if count==300:
    # if 300<=count<=400:
    if count==0:
        
        blank = np.ones((W,H),np.float32)
        a1=blank
        a=a1
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
    else:
        a=np.zeros((W, H),np.float32)
    # print(a[50,50])   
    #視細胞
        #a2 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    photoreceptor_cell = cv2.GaussianBlur(a, (7, 7), 1.0)
    #photoreceptor = (photoreceptor_cell * 255).astype(np.uint8)
    #水平細胞
    horizontal_cell= cv2.GaussianBlur(photoreceptor_cell,(19, 19), 3.0)
    horizontal = horizontal_cell_old*alpha_h+horizontal_cell*(1-alpha_h)
    horizontal_cell_old =  horizontal
    
    #双極細胞
    # bipolar_cell = cv2.absdiff(photoreceptor_cell, horizontal)    
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
    # np.savetxt('Test.csv',sum_b2,fmt='%12.3f' )
    
    # 半波整流
    hk=sum_b2>0
    hk1=hk*sum_b2
    b_on=hk1
    g=b_on[50,50]
    AC=b_on_old1*alpha_ac1+b_on*(1-alpha_ac1)
    AC1=b_on_old2*alpha_ac2+b_on*(1-alpha_ac2)
    # AC2=k_b2*AC+k_b3*AC1
    AC2=k_a1*AC+k_a2*AC1
    b_on_old1=AC
    b_on_old2=AC1
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
    tp=photoreceptor_cell[50,50]
    t_h=horizontal[50,50]
    t_b=sum_b[50,50]
    g=b_on[50:50]
    t_ac=AC2[50,50]
    # ax11.scatter(count, tp)
    # ax12.scatter(count, t_h)
    # ax13.scatter(count, t_b)
    # ax14.scatter(count, t_ac)
    s[i][1]=tp
    s[i][2]=t_h
    s[i][3]=t_b
    s[i][4]=t_ac
 
    np.savetxt('IIR_filter.csv',s , delimiter=",",fmt='%12.3f')
    # plt.plot(tp)
    # plt.plot(t_h)
    # plt.plot(t_b)
    # plt.plot(t_ac)
    
    # #スパイク発火
    # g1=g*255
    # # g2=0.0392*g1#i=10
    # # g2=0.0588*g1
    # g2=32*g1#i=15
    # if g2>=511.99:
    #     g2=511.99
    
    # g2=g1
    t_spike=count*dt
    # if t_spike<0:
    #     g2=0
    # elif 5*(2*i-2)<=t_spike<=5*(2*i-1):
    #     g2=15.875
    # elif 5*(2*i-1)<=t_spike<10*i:
    #     g2=-15.875
    # dv = (k_spike*(v - vrest)*(v - vthr) - u + I[i]) / C
    # dv = (k_spike*(v - vrest)*(v - vthr) - u + g2) / C
    # v = v + dt*dv # 膜電位の更新
    # u = u + dt*(a_spike*(b_spike*(v_-vrest)-u)) # 膜電位の更新
    # s = 1*(v>=vpeak) #発火時は 1, その他は 0 の出力
    # u = u + d_spike*s # 発火時に回復変数を上昇
    # v = v*(1-s) + vreset*s # 発火時に膜電位
    # v_ = v # v(t-1) <- v(t)
    # print(v)
    # print(u)
    # v_arr[i] = v # 膜電位の値を保存
    # u_arr[i] = u # 回復変数の値を保存
    # t = np.arange(nt)*dt
    # Line1=ax15.scatter(i, v_arr[i],color="b")
    # Line2=ax16.scatter(i, u_arr[i],color="b")
    # # s[i][0]=v_arr[i]
    # # s[i][1]=u
    # plt.plot(v_arr[i])
    # plt.plot(u_arr[i])
    # line1 = Line1.pop(0)
    # line2 = Line2.pop(0)
    # Line1.remove()
    # Line2.remove()

    # スパイク発火オイラー
    # v=v_old+(0.04*v*v+5*v+140-u+g2)*dt
    # v=v_old+(0.04*v_old*v_old+5*v_old+140-u_old+g2)*(dt/23)
    # u=u_old+a_spike*(b_spike*v_old-u_old)*dt
    # u=u_old+a_spike*(b_spike*v_old-u_old)*dt
    # u=a_spike*(b_spike*v_old-u_old)*dt

    # dv = ((v +60)*(v+40) - u + g2) / 23
    # v = v + dt*dv # 膜電位の更新
    # u = u + dt*(a_spike*(b_spike*(v_-vrest)-u)) # 膜電位の更新
    # s = 1*(v>=vpeak) #発火時は 1, その他は 0 の出力
    # u = u + d_spike*s # 発火時に回復変数を上昇
    # v = v*(1-s) + vreset*s # 発火時に膜電位
    # v_ = v # v(t-1) <- v(t)

    # if v>=30:
    #     v=c_spike
    #     # u=u_old+d_spike
    #     u=u+d_spike
    # v_old=v
    # u_old=u
    # v_arr[i] = v # 膜電位の値を保存
    # u_arr[i] = u# 回復変数の値を保存

    #空間フィルタ
    cs_p=photoreceptor_cell[50,:]
    cs_h=horizontal[50,:]
    # cs_b1=bipolar_cell[64,:]
    cs_b1=b_on[50,:]
    # ギャップ結合しているやつ
    cs_bon1=sum_b2[50,:]
    cs_bon=AC2[50,:]
    np.savetxt('phIIR.csv',cs_p , delimiter=",",fmt='%12.6f')
    np.savetxt('hIIR.csv',cs_h , delimiter=",",fmt='%12.6f')
    np.savetxt('bonIIR.csv',cs_b1 , delimiter=",",fmt='%12.6f')
    np.savetxt('sumb2IIR.csv',cs_bon , delimiter=",",fmt='%12.6f')
    np.savetxt('AC2IIR.csv',cs_bon1 , delimiter=",",fmt='%12.6f')
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
    count1[i] = count
    t = time.time() - start
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
print("finish")
# t1 = np.arange(nt)*dt
# np.savetxt('a.csv',v_arr , delimiter=",",fmt='%12.6f')
# np.savetxt('b.csv',u_arr , delimiter=",",fmt='%12.6f')
# # np.savetxt('b.csv',count1 , delimiter=",",fmt='%12.6f')
# plt.figure(figsize=(5, 5))
# plt.subplot(2,1,1)
# plt.plot(count1, v_arr, color="k")
# plt.ylabel('Membrane potential (mV)')
# plt.subplot(2,1,2)
# plt.plot(count1, u_arr, color="k")
# plt.xlabel('Time (ms)')
# plt.ylabel('Recovery current (pA)')
# plt.show()