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
    --> imresize()は単純にサンプル間の補間をしてるみたい(補間の方法は4種類 linear, cube など)

・ダウンサンプリングは単純に飛ばし飛ばしでサンプリングするだけ？（スケールファクタが実数であれば補間が必要？）

次回
・imresize（）でtest画像でアップサンプリングしてみる　ok 
・複素無線チャネルを実部と虚部をチャネル数の方向で分離して入れてみる <--- 次回 GANにチャネル入力してみる
    その前に
        画像入力でうまく補間ができるように調整する
        現段階で画像入力がうまくいっていないのは、正規化のせいでは？　正規化戻してabs()で画像出力してみる
    
・学習の詳細把握（損失関数の数式を追ってみる）
・CNNのチュートリアルやる（動画でもいいかも）
・GoogleColabでのGPU利用
・（SinGANはあまり時間かけずにその他の手法に力入れたほうがいいかも）

後でなおすところ
・保存済みモデルが削除されないように変更
    ・SR.py の ir2trained_model = ""
    ・training.py の27行目
"""