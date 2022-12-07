import cv2
import numpy as np
import random

# 正規分布から乱数を取得
#rnd_val = random.gauss(0,2)

def image_slide(img):

    # 画像サイズ
    height = img.shape[0]  # 高さ
    width  = img.shape[1]  # 幅

    # 平行移動する値を決定 画像をnpxずつ並行移動
    dx,dy = round(random.gauss(0,1)),round(random.gauss(0,1))

    # 平行移動の変換行列を作成
    afin_matrix = np.float32([[1,0,dx],[0,1,dy]])

    # アファイン変換適用
    afin_img = cv2.warpAffine(img,           # 入力画像
                            afin_matrix,   # 行列
                            (width,height)  # 解像度
                            )
    return afin_img

#height_em = 120
#width_em = 120
#random_pattern_pre=np.random.uniform(0, 1, (height_em,width_em))    
#random_pattern_pre_em=image_slide(random_pattern_pre)
#random_pattern_em=random_pattern_pre_em[10:110, 10:110]   
#a_append=[random_pattern_em]


for i in range(10):
    

    height_em = 120
    width_em = 120

    random_pattern_pre=np.random.uniform(0, 1, (height_em,width_em))    
    random_pattern_pre_em=image_slide(random_pattern_pre)
    random_pattern_em=random_pattern_pre_em[10:110, 10:110]
    visual_input=random_pattern_em

    if i==0:
        visual_input_append=[visual_input]
    else:
        visual_input_append = np.append(visual_input_append, [visual_input],axis=0) 

    cv2.imshow('eyemovement',visual_input)

    #繰り返し分から抜けるためのif文
    key =cv2.waitKey(1000)
    if key == 27:
        break

cv2.destroyAllWindows()
np.save('visual_input_append.npy',visual_input_append)







