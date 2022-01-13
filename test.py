import numpy as np
import torch

aaa = np.ones(5)
print(aaa)

x = torch.rand(5, 3)
print(x)

"""
Gs: Generator  (GeneratorConcatSkip2CleanAdd)
Zs: ノイズ? all 0
reals: 複数解像度の画像
NoiseAmp: ノイズの標準偏差 (z_in = noise_amp*(z_curr)+I_prev)

疑問
・元画像よりも解像度の高い画像をどうやって作ってるの？
・SinGAN_generate（）のなかでは結局、複数枚ストックしたreals_srは使わずに、reals_sr[0]だけ初期G入力として使ってるみたい
・imresize()の仕組みがわからない（これだけでupsamplinmgを済ませているの?）

次回
・imresize（）でのアップサンプル方法を調べる

"""