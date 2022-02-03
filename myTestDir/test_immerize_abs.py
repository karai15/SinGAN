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
    # 画像情報
    opt.input_dir = "./test_image/"
    # opt.input_name = "33039_LR.png"
    opt.input_name = "channel_obs_abs.png"
    opt.nc_im = 1  # imageのチャネル数
    r = 4  # upscale

    ##############################################
    # 画像取得
    image_ts = functions.read_image(opt)  # Tensor
    # upsampling
    image_up_ts = imresize(image_ts, r, opt)  # 1回目 (120, 80) -> (151, 101) (2回目以降さらに大きくなる)
    # npに変換
    image_np = functions.convert_image_np(image_ts)  # numpy
    image_up_np = functions.convert_image_np(image_up_ts)  # numpy

    # true 画像読み込み
    img_true = np.array(img.imread(opt.input_dir + "channel_true_abs.png")) # (120, 80, 3)
    img_true = img_true[:, :, 0] / 255  # [0 1] に正規化
    # img_obs = np.array(img.imread(opt.input_dir + "channel_obs_abs.png")) # (120, 80, 3)

    img_SinGAN = np.array(img.imread(opt.input_dir + "channel_obs_abs_HR.png"))
    img_SinGAN = np.average(img_SinGAN[:, :, 0:3], axis=2) / 255  # [0 1] に正規化


    #######################################
    # 読み込み
    data = np.load(opt.input_dir + '/channel_np_abs.npz')
    H_true = data["arr_0"]
    H_obs = data["arr_1"]
    #######################################


    #######################################
    # 元のスケールに戻す
    img_obs_den = denormalizaton(image_np, np.min(H_obs), np.max(H_obs))
    img_up_den = denormalizaton(image_up_np, np.min(H_obs), np.max(H_obs))
    img_SinGAN_den = denormalizaton(img_SinGAN, np.min(H_obs), np.max(H_obs))
    img_true_den = denormalizaton(img_true, np.min(H_true), np.max(H_true))

    #######################################
    # MSE評価
    N = H_true.size
    N_obs = H_obs.size
    MSE_up = np.sum(np.abs(img_true_den - img_up_den) ** 2) / N
    MSE_singan = np.sum(np.abs(img_true_den - img_SinGAN_den) ** 2) / N
    MSE_obs = np.sum(np.abs(H_obs - img_obs_den) ** 2) / N_obs
    print("MSE_up: ", MSE_up)
    print("MSE_singan: ", MSE_singan)
    print("MSE_obs: ", MSE_obs)
    #######################################


    #######################################
    # plot
    fig, axes = plt.subplots(1, 4, tight_layout=True, squeeze=False)
    axes[0, 0].imshow(img_true_den, cmap="jet")  # jet, Greys_r
    axes[0, 1].imshow(img_obs_den, cmap="jet")  #
    axes[0, 2].imshow(img_up_den, cmap="jet")  #
    axes[0, 3].imshow(img_SinGAN_den, cmap="jet")  #
    plt.show()
    #######################################



#     # SinGAN画像読み込み
#     # img_SinGAN = np.array(img.imread(opt.input_dir + "channel_obs_HR_alpha10.png"))
#     # img_SinGAN = np.array(img.imread(opt.input_dir + "channel_obs_HR_min11.png"))
#     # img_SinGAN = np.array(img.imread(opt.input_dir + "channel_obs_HR_min25.png"))
#
#     img_SinGAN = img_SinGAN[:, :, 0:3] / 255  # [0 1] に正規化
#
#     # [0 1] -> 元のスケール
#     img_complex = denormalizaton_complex(image_np, min_H_obs, max_H_obs)
#     img_up_complex = denormalizaton_complex(image_up_np, min_H_obs, max_H_obs)
#     img_true_complex = denormalizaton_complex(img_true, min_H, max_H)
#     img_SinGAN_complex = denormalizaton_complex(img_SinGAN, min_H_obs, max_H_obs)
#
#
#     ######################################
#     # test (画像化に伴う量子化誤差が大きすぎるかも)
#     e_obs = np.sum(np.abs(H_obs - img_complex)) / H_obs.size
#     e_obs_up = np.sum(np.abs(H_obs_up - img_up_complex)) / H_true.size
#     e_true = np.sum(np.abs(H_true - img_true_complex)) / H_true.size
#     e_upsample = np.sum(np.abs(H_true - img_up_complex)) / H_true.size
#     e_singan = np.sum(np.abs(H_true - img_SinGAN_complex)) / H_true.size
#     ######################################
#
#     ##############################################
#     # plot
#     fig, axes = plt.subplots(1, 4, tight_layout=True, squeeze=False)
#     axes[0, 0].imshow(np.abs(img_complex), cmap="jet")
#     axes[0, 0].set_title("obs")
#     axes[0, 1].imshow(np.abs(img_up_complex), cmap="jet")
#     axes[0, 1].set_title("upsampling")
#     axes[0, 2].imshow(np.abs(img_SinGAN_complex), cmap="jet")
#     axes[0, 2].set_title("SinGAN")
#     axes[0, 3].imshow(np.abs(img_true_complex), cmap="jet")
#     axes[0, 3].set_title("true")
#     plt.suptitle("Image")
#     ##############################################
#
#     plt.show()
#
#

# [0 1] に正規化された行列を元のスケールに戻す
def denormalizaton(H, min_H, max_H):
    return (max_H - min_H) * H + min_H


#
# # チャネルを画像保存できる形式に変換
# def create_save_H(H):
#     H, min_H, max_H = normalizaton_complex(H)  # [0 1] に正規化 (min_H, max_H は元のスケールに戻すときに必要)
#     H_save = np.zeros((H.shape[0], H.shape[1], 3))
#     H_save[:, :, 0] = np.real(H)  # (real, imag, zeros)
#     H_save[:, :, 1] = np.imag(H)
#     return H_save, min_H, max_H
#
# # 複素行列を [0 1] に正規化
# def normalizaton_complex(H):
#     min_H = np.min(np.array([np.min(np.real(H)), np.min(np.imag(H))]))  # 実部と虚部の最小値の中での最小値
#     max_H = np.max(np.array([np.max(np.real(H)), np.max(np.imag(H))]))  # 実部と虚部の最大値の中での最大値
#     H = (H - min_H * (1 + 1j)) / (max_H - min_H)  # [0 1]に正規化
#     return H, min_H, max_H
#
# # [0 1] に正規化された複素行列を元のスケールに戻す
# def denormalizaton_complex(H, min_H, max_H):
#     H = (max_H - min_H) * H + min_H
#     return H[:, :, 0] + 1j * H[:, :, 1]




if __name__ == '__main__':
    main()

