import yaml
from train.train_patchtst import run

if __name__ == "__main__":
    # 读取 YAML 配置文件并解析成字典
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 传递配置字典到 run 函数
    run(config)
