import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.patchtst import PatchTST
from utils.dataset import SineDataset
import yaml
from utils.dataset import ElectricityDataset
import matplotlib.pyplot as plt

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        output = model(x)  # [B, pred_len, C]
        output = output[..., -1]  # 只取 OT，对齐 y 的维度 [B, pred_len]

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)  # [B, pred_len, C]
            output = output[..., -1]  # 只取 OT

            loss = criterion(output, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def run(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集
    train_set = ElectricityDataset(config["data_path"], config["input_len"], config["pred_len"], split='train')
    val_set = ElectricityDataset(config["data_path"], config["input_len"], config["pred_len"], split='val')

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"])

    # 模型初始化
    model = PatchTST(
        input_len=config["input_len"],  # 输入长度
        pred_len=config["pred_len"],    # 预测长度
        patch_len=config["patch_len"],  # patch大小
        stride=config["stride"],        # 步幅
        d_model=config["d_model"],      # 模型维度
        n_heads=config["n_heads"],      # 多头注意力数
        d_ff=config["d_ff"],            # 前馈神经网络维度
        num_layers=config["num_layers"],# Transformer层数
        dropout=config["dropout"],      # Dropout率
        n_features=train_set.n_features # 特征数量（从数据集读取）
    ).to(device)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    # 训练过程
    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    plot_loss_curve(train_losses, val_losses)

if __name__ == "__main__":
    # 加载配置文件
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run(config)
