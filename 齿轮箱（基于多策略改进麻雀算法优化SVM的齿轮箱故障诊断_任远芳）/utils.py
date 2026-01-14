import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
import os
import pandas as pd

def vmd(signal, alpha=2000, tau=0.001, K=10, eps=1e-7, max_iter=500):
    """变分模态分解（VMD）实现（保持不变）"""
    N = len(signal)
    t = np.arange(N) / N
    f = np.fft.fftfreq(N, d=1/N)
    f_idx = np.argsort(f)

    u = np.zeros((K, N), dtype=np.complex_)
    omega = np.zeros(K)
    lambda_ = np.zeros(N, dtype=np.complex_)

    f_signal = np.fft.fft(signal)

    for iter in range(max_iter):
        for k in range(K):
            sum_other = np.sum(u[:k], axis=0) + np.sum(u[k+1:], axis=0)
            numerator = f_signal - sum_other + lambda_ / 2
            denominator = 1 + 2 * alpha * (f - omega[k])**2
            u[k] = np.fft.ifft(numerator / denominator)

        for k in range(K):
            u_fft = np.fft.fft(u[k])
            omega[k] = np.sum(f * np.abs(u_fft)**2) / np.sum(np.abs(u_fft)**2)

        sum_u = np.sum(u, axis=0)
        lambda_ = lambda_ + tau * (f_signal - np.fft.fft(sum_u))

        error = np.sum(np.linalg.norm(u[:, 1:] - u[:, :-1], axis=1)**2) / np.sum(np.linalg.norm(u[:, :-1], axis=1)**2)
        if error < eps:
            break

    u = np.real(u)
    return u, omega

def calculate_features(imf):
    """计算9维特征（保持不变）"""
    imf = np.array(imf)
    mean_val = np.mean(imf)
    var_val = np.var(imf)
    peak_val = np.max(np.abs(imf))
    kurtosis_val = signal.kurtosis(imf)
    rms_val = np.sqrt(np.mean(imf**2))
    
    peak_factor = peak_val / rms_val if rms_val != 0 else 0
    impulse_factor = peak_val / np.mean(np.abs(imf)) if np.mean(np.abs(imf)) != 0 else 0
    shape_factor = rms_val / np.mean(np.abs(imf)) if np.mean(np.abs(imf)) != 0 else 0
    clearance_factor = peak_val / (np.abs(kurtosis_val)**(1/4)) if kurtosis_val != 0 else 0

    return np.array([mean_val, var_val, peak_val, kurtosis_val, rms_val,
                     peak_factor, impulse_factor, shape_factor, clearance_factor])

def sliding_window(data, window_size=2048, step=1000):
    """滑动窗口划分（保持不变）"""
    n_samples = (len(data) - window_size) // step + 1
    samples = np.zeros((n_samples, window_size))
    for i in range(n_samples):
        samples[i] = data[i*step : i*step + window_size]
    return samples

def load_dataset(root_path, window_size=2048, step=1000, samples_per_class=200):
    """
    加载bearingset和gearset文件夹下的所有数据
    参数：
        root_path: 数据集根目录（包含bearingset和gearset文件夹）
        window_size: 滑动窗口大小（论文2048）
        step: 滑动步长（论文1000）
        samples_per_class: 每类故障抽取样本数（论文200）
    返回：
        X_train, X_test: 训练/测试特征（9维）
        y_train, y_test: 训练/测试标签（1-10，5类轴承+5类齿轮）
        label_map: 标签对应关系字典
    """
    # 定义故障类型与标签映射（轴承5类+齿轮5类，标签1-10）
    label_map = {
        # 轴承故障（bearingset）
        "health": 1,          # 正常运行
        "ball": 2,            # 滚子故障
        "inner": 3,           # 内圈故障
        "outer": 4,           # 外圈故障
        "combination": 5,     # 复合故障
        # 齿轮故障（gearset）
        "gear_health": 6,     # 正常运行
        "gear_chipped": 7,    # 缺损（裂纹）
        "gear_missing": 8,    # 断齿
        "gear_root": 9,       # 根部裂纹
        "gear_surface": 10    # 齿面磨损
    }

    X = []
    y = []

    # 遍历bearingset和gearset文件夹
    for folder in ["bearingset(1)", "gearset(1)"]:
        folder_path = os.path.join(root_path, folder)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"文件夹不存在：{folder_path}")
        
        # 遍历文件夹下所有CSV文件
        for csv_file in os.listdir(folder_path):
            if not csv_file.endswith(".csv"):
                continue
            file_path = os.path.join(folder_path, csv_file)
            
            # 1. 读取CSV文件（跳过配置信息行，只取数据行）
            df = pd.read_csv(file_path, skiprows=18)  # 跳过前18行配置信息（按数据示例调整）
            df = df.iloc[:, :8]  # 只保留前8列信号数据（去除可能的多余列）
            df = df.dropna()  # 删除空行
            signal_data = df.values  # 转换为numpy数组
            
            # 2. 选择信号通道（论文用第4列：行星齿轮z方向振动信号）
            target_signal = signal_data[:, 3]  # 第4列（索引3）为z方向振动信号
            
            # 3. 滑动窗口划分样本
            samples = sliding_window(target_signal, window_size=window_size, step=step)
            
            # 4. 确定当前文件的故障类型（通过文件名匹配）
            file_name = csv_file.lower()
            if folder == "bearingset(1)":
                if "health" in file_name:
                    label = label_map["health"]
                elif "ball" in file_name:
                    label = label_map["ball"]
                elif "inner" in file_name:
                    label = label_map["inner"]
                elif "outer" in file_name:
                    label = label_map["outer"]
                elif "combination" in file_name:
                    label = label_map["combination"]
                else:
                    print(f"跳过未知轴承故障文件：{csv_file}")
                    continue
            else:  # gearset
                if "health" in file_name:
                    label = label_map["gear_health"]
                elif "chipped" in file_name:
                    label = label_map["gear_chipped"]
                elif "missing" in file_name:
                    label = label_map["gear_missing"]
                elif "root" in file_name:
                    label = label_map["gear_root"]
                elif "surface" in file_name:
                    label = label_map["gear_surface"]
                else:
                    print(f"跳过未知齿轮故障文件：{csv_file}")
                    continue
            
            # 5. 抽取指定数量样本（避免样本过多）
            if len(samples) > samples_per_class:
                samples = samples[:samples_per_class]  # 取前200个样本
            elif len(samples) < samples_per_class:
                print(f"警告：{csv_file} 样本数不足{samples_per_class}，实际{len(samples)}个")
            
            # 6. VMD分解+特征提取
            for sample in samples:
                # VMD分解（K=10，论文确定）
                u, _ = vmd(sample, K=10)
                # 选择包络谱熵最小的IMF分量
                envelope_entropy = []
                for imf in u:
                    envelope = np.abs(signal.hilbert(imf))
                    prob = envelope / np.sum(envelope)
                    entropy = -np.sum(prob * np.log2(prob + 1e-10))
                    envelope_entropy.append(entropy)
                best_imf = u[np.argmin(envelope_entropy)]
                # 计算9维特征
                features = calculate_features(best_imf)
                X.append(features)
                y.append(label)
    

    # 转换为numpy数组
    X = np.array(X)
    y = np.array(y)
    print("=== 数据集读取校验 ===")
    print(f"特征数组X形状：{X.shape}")  # 正常应是(n,9)，n>0
    print(f"标签数组y形状：{y.shape}")  # 正常应是(n,)，n>0
    print(f"标签类别及样本数：{np.bincount(y)}")  # 正常应显示10个类别，每个类别样本数≥200
    if len(X) == 0 or len(y) == 0:
        raise ValueError("错误：未读取到任何样本！请检查CSV文件读取逻辑")
    # 划分训练集（60%）和测试集（40%），保持类别平衡
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.6, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, label_map