import matplotlib.pyplot as plt

# 1つ目のデータ
x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
y1 = [
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
    72.37591]

plt.figure(figsize=(10, 5.5))
plt.xticks(x1)
plt.ylim(0,100)
plt.plot(x1, y1, marker='o', linestyle='-', color='purple')
plt.xlabel("Prompt Depth")
plt.ylabel("H")
plt.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.95)
# plt.title("Graph 1")
# plt.legend()
plt.grid()
plt.savefig("office_home_promptdepth.png")

# 2つ目のデータ
x2 = [1, 2, 4, 8, 16]
y_clipfit = [35.05071, 35.62678, 38.45778, 41.27629, 43.7078]
y2_1 = [49.27949541, 58.23613912, 61.52834812, 62.9434211, 63.64341184]
y2_2 = [54.66391326, 61.45490754, 65.69350766, 67.39516352, 68.78291141]
y_lp = [
        37.37392618,
        36.98543,
        36.31005,
        34.50,
        31.11896
    ]
# y_clipfit = [
#     34.46855594,
#     34.58649103,
#     36.70331672,
#     38.89984556,
#     40.18092394
# ]
plt.figure(figsize=(10,5.5))
plt.plot(x2, y_lp, marker='o', linestyle='-', color='greenyellow', label="LP++")
plt.plot(x2, y_clipfit, marker='o', linestyle='-', color='lime', label="CLIPFit")
plt.plot(x2, y2_2, marker='o', linestyle='-', color='orange', label="Baseline")
plt.plot(x2, y2_1, marker='o', linestyle='-', color='purple', label="Ours")
plt.xticks(x2)
plt.xlabel("# Training Samples Per Domain")
plt.ylabel("H")
# plt.title("# Training Samples Per Domain")
plt.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.95)
plt.legend()
plt.grid()
plt.savefig("office_home_shots.png")

# 3つ目のデータ
x3 = [0.01, 0.1, 0.5, 1, 10, 20]
y3 = [63.10643337, 63.05826268, 63.57183142, 70.13014424, 71.24001696, 72.31769946]
plt.xticks([i for i in range(len(x3))], [str(x) for x in x3])
plt.figure(figsize=(10, 5.5))
plt.plot([str(x) for x in x3], y3, marker='o', linestyle='-', color='purple')
# plt.xscale("log")  # x軸を対数スケールにする
plt.xlabel("Loss Weight γ")
plt.ylabel("H")
plt.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.95)
#plt.title("")
# plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.savefig("office_home_lossweight.png")