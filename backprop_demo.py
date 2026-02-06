import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. 准备数据：模拟一张 5x5 的黑白图片，中间有一个「十字」
# 1 代表白色，0 代表黑色
input_image = torch.tensor(
    [
        [
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        ]
    ],
    dtype=torch.float32,
)

custom_filter = torch.tensor(
    [[[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]]], dtype=torch.float32
)
# 2. 定义模型：一个简单的 3x3 卷积层
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
print("初始权重", conv_layer.weight.data)
# with torch.no_grad(): # 手动修改参数时不需要计算梯度
#    conv_layer.weight.copy_(custom_filter)
print("设置权重", conv_layer.weight.data)
# 3. 定义损失函数（衡量差距）与优化器（负责参数更新）
criterion = nn.MSELoss()
optimizer = optim.SGD(conv_layer.parameters(), lr=0.1)  # 学习率设为 0.1
# 4. 训练过程：前向传播 -> 计算损失 -> 反向传播 -> 更新参数

print("--- 训练开始 ---")
for epoch in range(100):
    # A. 前向传播 (Forward Pass)
    output = conv_layer(input_image)

    # 假设我们的目标是让输出的中心值变成 10（模拟某种特征提取）
    target = torch.full_like(output, 10.0)
    loss = criterion(output, target)
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # B. 反向传播 (Backpropagation)
    # 这一步会计算 loss 对卷积核 9 个参数的梯度，并储存在 .grad 属性中
    optimizer.zero_grad()  # 先清空旧梯度
    # 反向传播会自动计算每个参数对 loss 的影响程度（梯度）
    loss.backward()

    if epoch % 10 == 0 and conv_layer.weight.grad is not None:
        print("当前卷积核的梯度 (Gradients):\n", conv_layer.weight.grad[0][0])

    # C. 参数更新 (Parameter Update)
    # 根据梯度，把卷积核的数值往减少 Loss 的方向移动一小步
    optimizer.step()
    if epoch % 10 == 0:
        print("更新后的卷积核权重 (Weights):\n", conv_layer.weight.data[0][0])
        print("-" * 30)
print("训练完成！卷积核已经根据反向传播的结果进行了调整。")
output = conv_layer(input_image)
print(torch.full_like(output, 10.0))
print("最终卷积核权重 (Weights):\n", conv_layer.weight.data[0][0])
plt.imshow(conv_layer.weight.data[0, 0], cmap="gray")
plt.colorbar()
plt.title("Visualized Filter Weights")
plt.savefig("trained_filter.png")
