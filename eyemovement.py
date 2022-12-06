import cv2
import numpy as np
import random

# 正規分布から乱数を取得
rnd_val = random.gauss(0,2)
img = cv2.imread("a.jpg")

def image_slide(img):

    # 画像サイズ
    height = img.shape[0]  # 高さ
    width  = img.shape[1]  # 幅

    # 平行移動する値を決定 画像をnpxずつ並行移動
    dx,dy = round(random.gauss(0,1)),round(random.gauss(0,2))

    # 平行移動の変換行列を作成
    afin_matrix = np.float32([[1,0,dx],[0,1,dy]])

    # アファイン変換適用
    afin_img = cv2.warpAffine(img,           # 入力画像
                            afin_matrix,   # 行列
                            (width,height)  # 解像度
                            )
    return afin_img
img1=image_slide(img)
cv2.imshow('im',img1)
print('test')
print('finish')