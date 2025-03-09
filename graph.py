# import matplotlib.pyplot as plt
# import numpy as np

# # データの定義
# data = {
#     "H": [70.46911828, 67.65348004, 70.33453106, 72.53208851, 72.83742199, 71.86299944,
#               67.80706977, 72.49659861, 73.02181473, 72.41564771, 69.54485583, 72.42855772],
#     "A": [79.98364971, 76.86611213, 80.49224822, 80.92052995, 80.90673217, 81.3437691,
#               74.88767159, 81.96909026, 81.439492, 81.31230037, 79.6450395, 80.63049538],
#     "F": [64.84095505, 66.19952154, 64.64102905, 67.54066029, 68.0641273, 66.35837025,
#               70.01296959, 66.8939543, 68.19447541, 67.16736336, 66.13915054, 67.57094945]
# }

# x = np.arange(1, 13)  # 横軸の値 (1〜12)

# # 各データごとに個別のグラフを作成し、保存
# for key, values in data.items():
#     plt.figure(figsize=(6, 4))
#     plt.plot(x, values, marker='o', linestyle='-', label=key)
#     plt.xlabel('Insert Layer', fontsize=12)
#     plt.ylabel(f'{key}', fontsize=12)
#     # plt.title(key)
#     plt.xticks(x)
#     # plt.legend()
#     plt.ylim(0, 100)
#     # plt.grid()
#     # plt.subplots_adjust(bottom=0.16)
#     plt.subplots_adjust(left=0.12, right=0.95, bottom=0.16, top=0.96)  # 左右の余白を減らす
#     plt.savefig(f"insert_annalysis_{key}.png")  # 画像ファイルとして保存
#     plt.show()
import matplotlib.pyplot as plt
import numpy as np

# データの定義
plt.rcParams["font.size"] = 15
x_values = np.array([0.001, 0.01, 0.1, 0.5, 1, 5, 10])  # X軸 (パラメータ)
x_v = np.array([i for i in range(1, 13)])
list3 = [
    69.72234141,
    66.78255237,
    70.60961593,
    72.39535992,
    72.71040913,
    72.31409298,
    64.83571745,
    73.03719356,
    73.0600055,
    72.50850409,
    71.58160661,
    72.37591423
]

y_v = np.array(list3)
x_values = np.array([1, 2, 4, 8, 16])
y_values_lpplus = np.array(
    [
        37.37392618,
        36.98543,
        36.31005,
        34.50,
        31.11896
    ]
)
y_values_clipfit = np.array([
    34.46855594,
    34.58649103,
    36.70331672,
    38.89984556,
    40.18092394
])
x_values_N = [1, 2, 3]
y_values_lp_N = [30.46, 31.11, 33.57]
y_values_clipfit_N = [43.44, 40.53, 50.02]
y_values_vpt_N = [59.89, 62.94, 67.40]
y_values_ours_N = [60.15, 67.40, 74.28]
y_values_vpt = np.array([49.27949541, 58.23613912, 61.52834812, 62.9434211, 63.64341184])
y_values_ours = np.array([54.66391326, 61.45490754, 65.69350766, 67.39516352, 68.78291141])
y_values = np.array([66.03826, 66.11853, 66.51367, 65.94166, 67.4, 73.14008, 74.23593])  # Y軸 (値)
x = np.array([i for i in range(len(x_values))])
# グラフの作成
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.1, top=0.95)
plt.figure(figsize=(10, 5.5))

# 軸ラベルの設定
# plt.xlabel("# Forgotten Domains")
plt.xlabel("prompt depth")
# plt.xlabel("loss weight γ")
# plt.xlabel("")
plt.ylabel("H")

# X軸を対数スケールに変更（0.001, 0.01, 0.1 のような値があるため）
# plt.xticks([i for i in range(len(x_values))], [str(x) for x in x_values])
plt.xticks([i for i in x_v])
# plt.xticks([i for i in x_values_N])
plt.ylim(0, 100)
# plt.plot(x_values, y_values_lpplus, marker="o",  linestyle='-', color="greenyellow",label="LP++")
# plt.plot(x_values, y_values_clipfit, marker="o",  linestyle='-', color="lime",label="CLIPFit")
# plt.plot(x_values, y_values_vpt, marker='o', linestyle='-', color="orange",label="Baseline")
# plt.plot(x_values, y_values_ours, marker='o', linestyle='-', color="purple",label="Ours")
# plt.plot(x_values_N, y_values_lp_N, marker="o",  linestyle='-', color="greenyellow",label="LP++")
# plt.plot(x_values_N, y_values_clipfit_N, marker="o",  linestyle='-', color="lime",label="CLIPFit")
# plt.plot(x_values_N, y_values_vpt_N, marker='o', linestyle='-', color="orange",label="Baseline")
# plt.plot(x_values_N, y_values_ours_N, marker='o', linestyle='-', color="purple",label="Ours")
plt.plot(x_v, y_v, marker="o", linestyle="-", color="purple")
# plt.plot(x, y_values, marker="o", linestyle="-", color="purple")

# 判例の表示
# plt.legend()

# グリッド表示
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.savefig(f"promp_depth.png")

# グラフの表示
plt.show()

