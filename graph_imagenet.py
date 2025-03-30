import matplotlib.pyplot as plt

# 1つ目のデータ
x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
y1 = [71.59626174, 73.97207696, 74.05963471, 74.04656722, 74.0842169, 
      74.35345377, 74.45369378, 74.85147828, 74.93466204, 75.06282545, 
      75.00361639, 74.23180563]

plt.figure(figsize=(10, 5.5))
plt.xticks(x1)
plt.plot(x1, y1, marker='o', linestyle='-', color='purple')
plt.xlabel("Prompt Depth")
plt.ylabel("H")
plt.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.95)
# plt.title("Graph 1")
# plt.legend()
plt.grid()
plt.savefig("imagenet_promptdepth.png")

# 2つ目のデータ
x2 = [1, 2, 4, 8, 16]
y_clipfit = [66.36249784, 68.94743613, 69.5784669, 70.60159937, 71.31357264]
y2_1 = [71.06660769, 72.6544234, 74.15602946, 75.3198087, 76.4368025]
y2_2 = [71.4862613, 73.22191489, 74.93466204, 78.37646254, 81.42303776]

plt.figure(figsize=(10,5.5))
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
plt.savefig("imagenet_shots.png")

# 3つ目のデータ
x3 = [0.01, 0.1, 0.5, 1, 5, 10, 20]
y3 = [74.8159675, 74.73645648, 74.92617869, 75.00361639, 
      75.08654942, 75.08265527, 75.05047032]
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
plt.savefig("imagenet_lossweight.png")