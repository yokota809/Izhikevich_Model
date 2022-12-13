import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import keyboard
import random
import cupy as cp

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
alpha_ac1=0.9
alpha_ac2=2*alpha_ac1-1
k_a1=2      
k_a2=-1
# 黒のサイズ
W=int(100)
H=int(100)
blank = np.zeros((W, H),np.float32)
blank1 = np.zeros((120, 120),np.float32)
# 白のサイズ
W1=int(20)
H1=int(20)
# 水平細胞の初期値
horizontal_cell_old=cp.ones((W, H),cp.float32)*0.5
# 双極細胞の初期値
bipolar_cell1=cp.ones((W, H),cp.float32)
bipolar_cell_old1=cp.ones((W, H),cp.float32)
bipolar_cell_old2=cp.ones((W, H),cp.float32)
bipolar_cell_old3=cp.ones((W, H),cp.float32)
# ON型の双極細胞
b_on_old1=cp.ones((W, H),cp.float32)
b_on_old2=cp.ones((W, H),cp.float32)
b_on=cp.ones((W, H),cp.float32)
# アマクリン細胞の初期値
AC2=cp.ones((W, H),cp.float32)
a=cp.ones((W, H),cp.float32)
b1=cp.ones((W, H),cp.float32)
b2=cp.ones((W, H),cp.float32)
b3=cp.ones((W, H),cp.float32)
count=0
# 眼球運動
def image_slide(img):

    # 画像サイズ
    height = img.shape[0]  # 高さ
    width  = img.shape[1]  # 幅

    # 平行移動する値を決定 画像をnpxずつ並行移動
    dx,dy = round(random.gauss(0,1)),round(random.gauss(0,1))

    # 平行移動の変換行列を作成
    afin_matrix = np.float32([[1,0,dx],[0,1,dy]])

    # アフィン変換適用
    afin_img = cv2.warpAffine(img,           # 入力画像
                            afin_matrix,   # 行列
                            (width,height)  # 解像度
                            )
    return afin_img

# スパイクパラメータ
dt = 0.1; T = 50 # ms
nt = round(T/dt) # シミュレーションステップ数
nt1 = round(T/dt)*10
count1=np.zeros(nt)
count1=np.zeros(nt)
t1 = np.arange(nt)*dt
a_spike = 0.02 # 回復時定数の逆数 (1/ms)
b_spike = 0.2 # u の v に対する共鳴度合い (pA/mV)
c_spike=-65
d_spike = 8 # 発火で活性化される正味の外向き電流 (pA)
vpeak = 30 #　ピーク電位 (mV)
# v_old=-65.75
v_old=-70.02
# u_old=7.76
u_old=-13.99
# CSVファイルに格納する配列
v_arr = cp.zeros(nt1,cp.float32) # 膜電位を記録する配列
u_arr = cp.zeros(nt1,cp.float32) # 回復変数を記録する配列
RF_sum = cp.zeros(nt1,cp.float32)
s=cp.zeros((nt,5), cp.float32)
spike_data=np.zeros((nt1,5), cp.float32)
sta_data=cp.zeros((W,H,nt), cp.float32)
# 半波整流
Rectify=cp.zeros((W, H),np.float32)

# スパイクニューロンモデルの入力のスケールパラメータ
scale_x=32
scale_gain=4
g_arr = np.zeros(nt1)
lnstart = time.time()

# 空間フィルターのグラフ設定
# fig1 = plt.figure(figsize=(9,6))
# plt.get_current_fig_manager().window.wm_geometry("+0+400")
# plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, hspace=0.2)
# plt.subplots_adjust(wspace=0.2, hspace=0.4)
# ph_fir = fig1.add_subplot(221, xlim=[0, 100], ylim=[0, 1.0], xlabel="x", ylabel="intensity[a.u.]")
# plt.title("PC")
# hc_fir = fig1.add_subplot(222, sharex=ph_fir, ylim=[0.48, 0.52], xlabel="x")
# plt.title("HC")
# bc_fir = fig1.add_subplot(223, sharex=ph_fir, ylim=[-0.02, 0.02], xlabel="x", ylabel="intensity[a.u.]")
# plt.title("BC")
# ac_fir = fig1.add_subplot(224, sharex=ph_fir, ylim=[-0.005, 0.01], xlabel="x")
# plt.title("AC")
# jiku=range(0, 100)

# 時間
start = time.time()

# スパイクの表示
# fig, ax = plt.subplots()
# fig = plt.figure
frames = []
# fig_spike = plt.figure(figsize=(9,6))
# plt.get_current_fig_manager().window.wm_geometry("+400+400")
# plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, hspace=0.2)
# plt.subplots_adjust(wspace=0.2, hspace=0.4)
# fig_input = fig_spike.add_subplot(221, xlim=[0,nt], ylim=[0, 1.0], xlabel="x", ylabel="intensity[a.u.]")
# plt.title("input")
# fig_impulse = fig_spike.add_subplot(222, sharex=fig_input, ylim=[0, 1], xlabel="x")
# plt.title("impulse")
# fig_v = fig_spike.add_subplot(223, sharex=fig_input, ylim=[-90, 40], xlabel="x", ylabel="intensity[a.u.]")
# plt.title("v")
# fig_responce = fig_spike.add_subplot(224, sharex=fig_input, ylim=[0, 1], xlabel="x")
# plt.title("responce")

# fig_input = fig_spike.add_subplot(221, ylim=[0, 1.0], xlabel="x", ylabel="intensity[a.u.]")
# plt.title("input")
# fig_impulse = fig_spike.add_subplot(222,ylim=[0, 1], xlabel="x")
# plt.title("impulse")
# fig_v = fig_spike.add_subplot(223,  ylim=[-90, 40], xlabel="x", ylabel="intensity[a.u.]")
# plt.title("v")
# fig_responce = fig_spike.add_subplot(224,  ylim=[0, 1], xlabel="x")
# plt.title("responce")

# 細胞応答
fig = plt.figure(figsize=(9,6))
plt.get_current_fig_manager().window.wm_geometry("+1000+300")
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, hspace=0.2)
# 余白を設定
# plt.yticks(np.arange(-1.0,1.0, step=0.2))

# 時間フィルタ
plt.subplots_adjust(wspace=0.2, hspace=0.4)
# ax11 = fig.add_subplot(221, xlim=[0, nt/4], ylim=[-1.5, 1.5], xlabel="frame number", ylabel="intensity[a.u.]")
# plt.title("PC")
# ax12 = fig.add_subplot(222, xlim=[0, nt/4], ylim=[0, 0.05], xlabel="frame number", ylabel="intensity[a.u.]")
# plt.title("HC")
# ax13 = fig.add_subplot(223, xlabel="frame number", ylabel="intensity[a.u.]")
# plt.title("BC")
# ax14 = fig.add_subplot(224, xlim=[0, nt/4], ylim=[-0.05, 0.05], xlabel="frame number", ylabel="intensity[a.u.]")
# plt.title("AC")

# ax11 = fig.add_subplot(221,  xlabel="frame number", ylabel="v")
# plt.title("V")
# ax12 = fig.add_subplot(222,  xlabel="frame number")
# plt.title("HC")
# ax13 = fig.add_subplot(223, xlabel="frame number", ylabel="intensity[a.u.]")
# plt.title("BC")
# ax14 = fig.add_subplot(224,  xlabel="frame number")
# plt.title("AC")

# 眼球運動用の画像サイズ
height_em = 120
width_em = 120
count1=0
Switch=0
# 入力選択
print("Do you include eyemovement?y/n")
# 眼球運動なしの場合
if keyboard.read_key() == "a":
    print("without eyemovement time impulse")
    Switch=1
    print(Switch)
elif keyboard.read_key() == "b":
    print("without eyemovement random impulse")
    Switch=2
    print(Switch)

# 眼球運動ありの場合
elif keyboard.read_key() == "c":
    print("with eyemovement time impulse")
    Switch=3
    print(Switch)
elif keyboard.read_key() == "d":
    print("with eyemovement random impulse")
    Switch=4
    print(Switch)
        
for i in range(0, nt):
    # 眼球運動考慮なし状態の矩形入力
    if Switch==1:
        if i==0:
            blank[40:60,40:60] = np.ones((W1, H1),np.float32)
            visual_input= blank  
        else:
            blank = np.zeros((W, H),np.float32)
            visual_input= blank
    # 眼球運動考慮なし状態のランダム入力
    elif Switch==2:
        visual_input=np.random.uniform(0, 1, (H,W))
    # 眼球運動考慮あり状態の矩形入力
    elif Switch==3:
        if i==0:
            blank1[60:80,60:80] = np.ones((W1, H1),np.float32)
            random_pattern_pre_em=image_slide(blank1)
            random_pattern_em=random_pattern_pre_em[10:110, 10:110]
            visual_input=random_pattern_em 
        elif not i==0 and i%3==0:
            blank1 = np.zeros((120, 120),np.float32)
            random_pattern_em=blank1[10:110, 10:110]
            visual_input=random_pattern_em 
        else:
            blank1 = np.zeros((120, 120),np.float32)
            random_pattern_pre_em=image_slide(blank1)
            random_pattern_em=random_pattern_pre_em[10:110, 10:110]
            visual_input=random_pattern_em 
    # 眼球運動考慮あり状態のランダム入力
    elif Switch==4:
        if not i==0 and i%3==0:
            random_pattern_pre=np.random.uniform(0, 1, (height_em,width_em))    
            random_pattern_pre_em=image_slide(random_pattern_pre)
            random_pattern_em=random_pattern_pre_em[10:110, 10:110]
            visual_input=random_pattern_em 
        else:
            random_pattern_pre=np.random.uniform(0, 1, (height_em,width_em))    
            random_pattern_em=random_pattern_pre[10:110, 10:110]
            visual_input=random_pattern_em
  
    if i==0:
        visual_input_append=[visual_input]
    else:
        visual_input_append = np.append(visual_input_append, [visual_input],axis=0) 
    
    # 視細胞の空間処理
    photoreceptor_cell = cv2.GaussianBlur(visual_input, (5, 5), 1.0)
    horizontal_cell_cp = cp.asarray(photoreceptor_cell)
        
    #水平細胞_cp演算
    horizontal_cell = cv2.GaussianBlur(photoreceptor_cell,(19, 19), 3.0)
    horizontal_cell_cp = cp.asarray(horizontal_cell)
    horizontal = horizontal_cell_old*alpha_h+horizontal_cell_cp*(1-alpha_h)
    horizontal_cell_old =  horizontal
    
    #双極細胞
    bipolar_cell = horizontal_cell_cp-horizontal
    b11=bipolar_cell_old1*alpha_a1+bipolar_cell*(1-alpha_a1)
    b12=bipolar_cell_old2*alpha_a2+bipolar_cell*(1-alpha_a2)
    b13=bipolar_cell_old3*alpha_a3+bipolar_cell*(1-alpha_a3)
    sum_b=k_b1*b11+k_b2*b12+ k_b3*b13
    bipolar_cell_old1=b11
    bipolar_cell_old2=b12
    bipolar_cell_old3=b13
    sum_b_np = cp.asnumpy(sum_b)
    
    #ギャップ結合した双極細胞とアマクリン細胞の抑制
    sum_b1 = cv2.GaussianBlur(sum_b_np, (7, 7), 1.0)
    sum_b_cp = cp.asarray(sum_b1)
    sum_b2=sum_b_cp-AC2
    cs_b=sum_b2[50,:]

    # 半波整流
    Rectify=sum_b2>0
    Rectify1=Rectify*sum_b2
    b_on=Rectify1
    b_on_np = cp.asnumpy(b_on)
    sum_b_on = cv2.boxFilter(b_on_np,ddepth=-1,ksize=(7,7),normalize=False)
    input_g=sum_b_on[50,50]

    # アマクリン細胞
    AC=b_on_old1*alpha_ac1+b_on*(1-alpha_ac1)
    AC1=b_on_old2*alpha_ac2+b_on*(1-alpha_ac2)
    AC2=k_a1*AC+k_a2*AC1
    b_on_old1=AC
    b_on_old2=AC1
    input_g_cp = cp.asarray(input_g)
    SpikingNeuron_input=input_g*scale_gain*scale_x
    
    # スパイク生成処理
    for j  in range(0,10):

        v=v_old+(0.04*v_old*v_old+5*v_old+140-u_old+SpikingNeuron_input)*(dt)
        u=u_old+a_spike*(b_spike*v_old-u_old)*dt
        if v>=vpeak:
            # print(v)
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
        # fig_input.scatter(count1, spike_data[i*10+j,0],color="b")
        # fig_impulse.scatter(count1, spike_data[i*10+j,1],color="b")
        # fig_v.scatter(count1, spike_data[i*10+j,2],color="b")
        # fig_responce.scatter(count1, spike_data[i*10+j,3],color="b")
        
        # グラフ
        # ax11.scatter(count1, spike_data[i*10+j,2],color="b")
        count1+=1
        # ani = animation.FuncAnimation(fig, frames, interval=200)
        # plt.show()
    # スパイク応答表示
    t_spike = np.arange(10)
    # t_spike = np.arange(nt)
    # input_Line=fig_input.plot(t_spike, spike_data[i*10:i*10+10,0],color="b")
    # impulse_Line2=fig_impulse.plot(t_spike, spike_data[i*10:i*10+10,1],color="b")
    # v_Line3=fig_v.plot(t_spike, spike_data[i*10:i*10+10,2],color="b")
    # responce_Line4=fig_responce.plot(t_spike, spike_data[i*10:i*10+10,3],color="b")
    # artists = ax.plot(t_spike, spike_data[i*10:i*10+10,2], c="b")
    
    # 時間特性
    # t_p=photoreceptor_cell[50,50]
    t_p=horizontal_cell_cp[50,50]
    # t_h=sum_b1[50,50]
    t_h=sum_b_cp[50,50]
    # t_h=horizontal[50,50]
    t_b=sum_b2[50,50]
    t_ac=AC2[50,50]
    s[i][0]=t_p
    s[i][1]=t_h
    s[i][2]=t_b
    s[i][3]=t_ac
    s[i][4]=SpikingNeuron_input
    # ax11.scatter(count, t_p,color="b")
    
    # ax12.scatter(count, t_h,color="b")
    # ax13.scatter(count, t_b,color="b")
    # ax14.scatter(count, t_ac,color="b")
    # np.savetxt('IIR_filter.csv',s , delimiter=",",fmt='%12.6f')
    #空間フィルタ処理後の表示
    cs_p=photoreceptor_cell[50,:]
    cs_h=horizontal[50,:]
    # ギャップ結合しているやつ
    cs_bon1=sum_b1[50,:]
    cs_ac=AC2[50,:]
    # Line1=ph_fir.plot(jiku, cs_p,color="b")
    # Line2=hc_fir.plot(jiku, cs_h,color="b")
    # Line3=bc_fir.plot(jiku, cs_bon1,color="b")
    # Line4=ac_fir.plot(jiku, cs_ac,color="b")
    
    # 画像の表示期間と除外
    plt.pause(0.01)
    # Input_Line=ani.pop(0)
    # Input_Line=input_Line.pop(0)
    # Impulse_Line2=impulse_Line2.pop(0)
    # V_Line3=v_Line3.pop(0)
    # Responce_Line4=responce_Line4.pop(0)  
    # line1 = Line1.pop(0)
    # line2 = Line2.pop(0)
    # line3 = Line3.pop(0)
    # line4 = Line4.pop(0)
    # Input_Line.remove()
    # Impulse_Line2.remove()
    # V_Line3.remove()
    # Responce_Line4.remove()
    # line1.remove()
    # line2.remove()
    # line3.remove()
    # line4.remove()
    
    count +=1
t = time.time() - start
print('t',t)
print("finish")
cp.savetxt('IIR_filter.csv',s , delimiter=",",fmt='%12.6f')
# np.save('visual_input_append1208.npy',visual_input_append)
# np.save('spike_data1208.npy',spike_data)
# np.save('IIR_data1208.npy',s)
t1= np.arange(nt1)*dt
# np.savetxt('Spike.csv',spike_data , delimiter=",",fmt='%12.6f')
# numpに変換
g_arr1 = cp.asnumpy(g_arr)
v_arr1 = cp.asnumpy(g_arr)
RF_sum1 = cp.asnumpy(RF_sum)
plt.figure(figsize=(9, 7))
fig.savefig("graph.png")
plt.subplot(3,1,1)
plt.scatter(t1, g_arr1, color="b")
plt.plot(t1, g_arr1, color="k")
plt.ylabel('spike_input')
plt.subplot(3,1,2)
plt.plot(t1, v_arr1, color="k")
plt.xlabel('Time (ms)')
plt.ylabel('v')
plt.subplot(3,1,3)
plt.plot(t1,RF_sum1, color="k")
plt.xlabel('Time (ms)')
plt.ylabel('spike_response')
plt.show()