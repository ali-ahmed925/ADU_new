import matplotlib.pyplot as plt
import numpy as np

# データの定義
data = {
    "H": [70.46911828, 67.65348004, 70.33453106, 72.53208851, 72.83742199, 71.86299944,
              67.80706977, 72.49659861, 73.02181473, 72.41564771, 69.54485583, 72.42855772],
    "A": [79.98364971, 76.86611213, 80.49224822, 80.92052995, 80.90673217, 81.3437691,
              74.88767159, 81.96909026, 81.439492, 81.31230037, 79.6450395, 80.63049538],
    "F": [64.84095505, 66.19952154, 64.64102905, 67.54066029, 68.0641273, 66.35837025,
              70.01296959, 66.8939543, 68.19447541, 67.16736336, 66.13915054, 67.57094945]
}

x = np.arange(1, 13)  # 横軸の値 (1〜12)

# 各データごとに個別のグラフを作成し、保存
for key, values in data.items():
    plt.figure(figsize=(6, 4))
    plt.plot(x, values, marker='o', linestyle='-', label=key)
    plt.xlabel('Insert Layer', fontsize=12)
    plt.ylabel(f'{key}', fontsize=12)
    # plt.title(key)
    plt.xticks(x)
    # plt.legend()
    plt.ylim(0, 100)
    # plt.grid()
    # plt.subplots_adjust(bottom=0.16)
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.16, top=0.96)  # 左右の余白を減らす
    plt.savefig(f"insert_annalysis_{key}.png")  # 画像ファイルとして保存
    plt.show()
