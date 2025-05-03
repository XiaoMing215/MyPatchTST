import numpy as np
import pandas as pd
import os

def generate_sine_dataset(seq_len=120, n_samples=500, n_features=3):
    X = []
    for _ in range(n_samples):
        t = np.linspace(0, 10, seq_len)
        series = np.stack([
            np.sin(t + np.random.rand() * 2 * np.pi) + 0.1 * np.random.randn(seq_len)
            for _ in range(n_features)
        ], axis=1)  # shape: [seq_len, n_features]
        X.append(series)
    return np.array(X)  # shape: [n_samples, seq_len, n_features]

if __name__ == "__main__":
    save_dir = ""
    os.makedirs(save_dir, exist_ok=True)

    data = generate_sine_dataset(seq_len=120, n_samples=500, n_features=3)

    # 展平成 DataFrame 保存为 CSV
    flat_data = []
    for i in range(data.shape[0]):
        sample = data[i]  # [120, 3]
        df = pd.DataFrame(sample, columns=[f"feature_{j}" for j in range(data.shape[2])])
        df["sample_id"] = i
        df["timestep"] = np.arange(data.shape[1])
        flat_data.append(df)

    all_data = pd.concat(flat_data, ignore_index=True)
    all_data.to_csv(f"{save_dir}/sine_data.csv", index=False)
    print(f"Saved shape: {all_data.shape}, path: {save_dir}/sine_data.csv")
