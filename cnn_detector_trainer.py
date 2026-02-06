from pyexpat import model
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math


def plot_epoch_snapshot(fig, rows, cols, plot_idx, epoch, model, output, loss):
    row_base = (plot_idx // cols) * 2
    col_pos = (plot_idx % cols) + 1

    # --- A. 绘制卷积核并标注数值 ---
    ax_w = fig.add_subplot(rows, cols, row_base * cols + col_pos)
    weight_data = model.conv.weight.data.squeeze().numpy()
    ax_w.imshow(weight_data, cmap="viridis")
    ax_w.set_title(f"E{epoch} Kernel Weights")
    ax_w.axis("off")
    # 遍历格点标注数值
    for i in range(3):
        for j in range(3):
            ax_w.text(
                j,
                i,
                f"{weight_data[i, j]:.4f}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
            )

    # --- B. 绘制特征图并标注数值 ---
    ax_o = fig.add_subplot(rows, cols, (row_base + 1) * cols + col_pos)
    output_data = output.data.squeeze().numpy()
    ax_o.imshow(output_data, cmap="magma", vmin=0, vmax=10)
    ax_o.set_title(f"E{epoch} Output Feature Map\nLoss: {loss.item():.2f}")
    ax_o.axis("off")
    # 遍历格点标注数值
    for i in range(3):
        for j in range(3):
            ax_o.text(
                j,
                i,
                f"{output_data[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
            )


def plot_test_snapshot(fig, rows, cols, num_plots, input_img, output_test):
    # 计算测试输出的行列位置
    test_col = (num_plots - 1) % cols + 1
    test_row_base = ((num_plots - 1) // cols) * 2

    # TEST Input
    ax_t_in = fig.add_subplot(rows, cols, test_row_base * cols + test_col)
    ax_t_in.imshow(input_img.squeeze().numpy(), cmap="gray")
    ax_t_in.set_title("TEST Input", color="red")
    ax_t_in.axis("off")

    # TEST Output 并标注数值
    ax_t_out = fig.add_subplot(rows, cols, (test_row_base + 1) * cols + test_col)
    test_out_data = output_test.squeeze().numpy()
    ax_t_out.imshow(test_out_data, cmap="magma", vmin=0, vmax=10)
    ax_t_out.set_title(f"TEST Output\nMax: {output_test.max():.2f}", color="red")
    ax_t_out.axis("off")
    for i in range(3):
        for j in range(3):
            ax_t_out.text(
                j,
                i,
                f"{test_out_data[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
            )

# 1. 数据准备
input_img = torch.tensor(
    [[[
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ]]], dtype=torch.float32)

# 2. 模型与目标
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, bias=False)

    def forward(self, x):
        return self.conv(x)
model = SimpleCNN() 
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
target = torch.tensor(
    [[[
        [0, 0, 0],
        [0, 9, 0],
        [0, 0, 0]
    ]]], dtype=torch.float32)

# 3. 定义观察点,每10个epoch记录一次，前10个epoch都记录
display_epochs = list(range(1, 11)) + list(range(20, 101, 10))
num_plots = len(display_epochs) + 1
cols = 5
rows = math.ceil(num_plots / cols) * 2
fig = plt.figure(figsize=(20, rows * 3.5))  # 稍微加大画布以容纳文字

plot_idx = 0
for epoch in range(1, 101):
    # 梯度清零 (Zero Gradients)
    optimizer.zero_grad()
    # 前向传播 (Forward Pass)
    output = model(input_img)
    # 计算损失 (Loss Calculation)
    loss = criterion(output, target)
    # 反向传播 (Backpropagation)
    loss.backward()
    # 参数更新 (Parameter Update)
    optimizer.step()

    if epoch in display_epochs:
        plot_epoch_snapshot(fig, rows, cols, plot_idx, epoch, model, output, loss)

        plot_idx += 1

# 4. 多个十字测试
non_cross_img = torch.tensor(
    [[[
        [0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 0],
    ]]], dtype=torch.float32)
# 测试输入并标注数值
with torch.no_grad():
    output_test = model(non_cross_img)
print("TEST Output:\n", output_test.squeeze().numpy())
plot_test_snapshot(fig, rows, cols, num_plots, non_cross_img, output_test)

plt.tight_layout()
plt.suptitle("CNN Evolution : Backpropagation Optimization and Feature Extraction", fontsize=20, y=1.0)
plt.subplots_adjust(top=0.95)  # 调整主标题位置
plt.savefig("cnn_evolution_with_labels.png")
# plt.show()
