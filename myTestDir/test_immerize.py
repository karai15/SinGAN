import sys
import os
sys.path.append("../")
import argparse
from skimage import io as img
import numpy as np

# 自作モジュールからのimport
from config import get_arguments
from SinGAN.imresize import imresize
import SinGAN.functions as functions
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.not_cuda = 1  # GPU利用なし

    ##############################################
    # 画像情報
    opt.input_dir = "./test_image/"
    # opt.input_name = "33039_LR.png"
    opt.input_name = "channel_obs.png"
    opt.nc_im = 3  # imageのチャネル数

    # 画像取得
    image_ts = functions.read_image(opt)  # Tensor

    # upsampling
    r = 4
    image_up_ts = imresize(image_ts, r, opt)  # 1回目 (120, 80) -> (151, 101) (2回目以降さらに大きくなる)

    # npに変換
    image_np = functions.convert_image_np(image_ts)  # numpy
    image_up_np = functions.convert_image_np(image_up_ts)  # numpy

    # true画像読み込み
    img_true = np.array(img.imread(opt.input_dir + "channel_true.png")) # (120, 80, 3)
    img_true = img_true[:, :, 0:3]

    # plot img
    fig, axes = plt.subplots(1, 3, tight_layout=True, squeeze=False)
    axes[0, 0].imshow(image_np, cmap="jet")
    axes[0, 0].set_title("origin")
    axes[0, 1].imshow(image_up_np, cmap="jet")
    axes[0, 1].set_title("upsampling")
    axes[0, 2].imshow(img_true, cmap="jet")
    axes[0, 2].set_title("true")
    plt.suptitle("Image")
    ##############################################

    ##############################################
    # データ読み込み
    data = np.load(opt.input_dir + '/channel_np.npz')
    H_true = data["arr_0"]
    H_obs = data["arr_1"]
    H_itpl = data["arr_2"]

    # upsampling
    H_obs_img, min_H_obs, max_H_obs = create_save_H(H_obs)  # numpy [0 1]に正規化
    H_obs_ts = functions.np2torch(H_obs_img * 255, opt)  # tensor  [0 255]で入力する必要
    H_obs_up_ts = imresize(H_obs_ts, r, opt)  # upsampling
    H_obs_up = functions.convert_image_np(H_obs_up_ts)  # numpy
    H_obs_up = denormalizaton_complex(H_obs_up, min_H_obs, max_H_obs)  # [0 1] -> 元のスケール

    # plot data upsample
    fig, axes = plt.subplots(1, 4, tight_layout=True, squeeze=False)
    axes[0, 0].imshow(np.abs(H_obs), cmap="jet")
    axes[0, 0].set_title("obs")
    axes[0, 1].imshow(np.abs(H_obs_up), cmap="jet")
    axes[0, 1].set_title("upsample")
    axes[0, 2].imshow(np.abs(H_itpl), cmap="jet")
    axes[0, 2].set_title("DFT interplation")
    axes[0, 3].imshow(np.abs(H_true), cmap="jet")
    axes[0, 3].set_title("true")
    plt.suptitle("H")
    ##############################################

    plt.show()



# チャネルを画像保存できる形式に変換
def create_save_H(H):
    H, min_H, max_H = normalizaton_complex(H)  # [0 1] に正規化 (min_H, max_H は元のスケールに戻すときに必要)
    H_save = np.zeros((H.shape[0], H.shape[1], 3))
    H_save[:, :, 0] = np.real(H)  # (real, imag, zeros)
    H_save[:, :, 1] = np.imag(H)
    return H_save, min_H, max_H

# 複素行列を [0 1] に正規化
def normalizaton_complex(H):
    min_H = np.min(np.array([np.min(np.real(H)), np.min(np.imag(H))]))  # 実部と虚部の最小値の中での最小値
    max_H = np.max(np.array([np.max(np.real(H)), np.max(np.imag(H))]))  # 実部と虚部の最大値の中での最大値
    H = (H - min_H * (1 + 1j)) / (max_H - min_H)  # [0 1]に正規化
    return H, min_H, max_H

# [0 1] に正規化された複素行列を元のスケールに戻す
def denormalizaton_complex(H, min_H, max_H):
    H = (max_H - min_H) * H + min_H
    return H[:, :, 0] + 1j * H[:, :, 1]

if __name__ == '__main__':
    main()


# def plot_image(Y, N_show, title):
#     """
#     :param Y: (N, L, L)
#     :param N_show:
#     """
#     # Observation
#     fig, axes = plt.subplots(N_show, N_show, tight_layout=True, squeeze=False)
#     cnt = 0
#     for i in range(N_show):
#         for j in range(N_show):
#             axes[i, j].imshow(Y[cnt, :, :])
#             cnt += 1
#     plt.suptitle(title)


##############################
# def read_image(opt):
#     x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))  # (120, 80, 3)
#     x = np2torch(x,opt)  # (1, 3, 120, 80)
#     x = x[:,0:3,:,:]  # 3チャネルだけ抽出
#     return x
#
# def imresize(im,scale,opt):
#     #s = im.shape
#     im = torch2uint8(im)  # [0 255]に変換
#     im = imresize_in(im, scale_factor=scale)  # upsampling
#     im = np2torch(im,opt)
#     #im = im[:, :, 0:int(scale * s[2]), 0:int(scale * s[3])]
#     return im
##############################
