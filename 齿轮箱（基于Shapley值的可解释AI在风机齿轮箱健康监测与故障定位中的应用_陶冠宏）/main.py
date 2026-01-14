import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from natsort import natsorted
from xai_framework import XAI_WindTurbine  # 导入之前的模型框架
from load_seu_gearbox_data import load_seu_gearbox_data, plot_time_domain_signal


# ---------------------- 2. 从原始数据提取特征，生成模型所需的fault_df ----------------------
def extract_features(csv_acc_data):
    if len(csv_acc_data) == 0:
        print("错误：无数据可提取特征！")
        return pd.DataFrame()
    features = []
    labels = []
    load_configs = []  # 记录负载配置（可选：作为特征）

    for item in csv_acc_data:
        data = item['data']
        folder = item['sub_folder']
        file_name = item['file_name'].lower()  # 小写处理，避免大小写问题
        load_config = item['load_config']

        # ---------------------- 提取有效信号特征 ----------------------
        # 有效信号：行星齿轮箱x/y/z（索引1/2/3）、并联齿轮箱x/y/z（索引5/6/7）
        # 示例：计算各方向振动的均值作为特征
        planet_x = data[:, 1]  # 行星齿轮箱x
        planet_y = data[:, 2]  # 行星齿轮箱y
        planet_z = data[:, 3]  # 行星齿轮箱z
        parallel_x = data[:, 5]  # 并联齿轮箱x
        parallel_y = data[:, 6]  # 并联齿轮箱y
        parallel_z = data[:, 7]  # 并联齿轮箱z

        # 特征：6个方向的均值+标准差（共12个特征）
        feature = [
            np.mean(planet_x), np.std(planet_x),
            np.mean(planet_y), np.std(planet_y),
            np.mean(planet_z), np.std(planet_z),
            np.mean(parallel_x), np.std(parallel_x),
            np.mean(parallel_y), np.std(parallel_y),
            np.mean(parallel_z), np.std(parallel_z)
        ]

        # ---------------------- 标注标签（共10类：轴承5类+齿轮5类） ----------------------
        if 'bearingset' in folder:
            # 轴承故障类型（5类）
            if 'health' in file_name:
                label = 0  # 正常运行
            elif 'ball' in file_name:
                label = 1  # 滚子故障
            elif 'inner' in file_name:
                label = 2  # 内圈故障
            elif 'outer' in file_name:
                label = 3  # 外圈故障
            elif 'comb' in file_name:
                label = 4  # 复合故障
            else:
                label = -1  # 未知类型
        elif 'gearset' in folder:
            # 齿轮故障类型（5类）
            if 'health' in file_name:
                label = 5  # 正常运行
            elif 'chipped' in file_name:
                label = 6  # 缺损（裂纹）
            elif 'miss' in file_name:
                label = 7  # 断齿
            elif 'root' in file_name:
                label = 8  # 根部裂纹
            elif 'surface' in file_name:
                label = 9  # 齿面磨损
            else:
                label = -1  # 未知类型
        else:
            label = -1

        # 过滤未知类型
        if label != -1:
            features.append(feature)
            labels.append(label)
            load_configs.append(load_config)  # 可选：保留负载配置

    # 生成DataFrame
    feature_cols = [
        'planet_x_mean', 'planet_x_std',
        'planet_y_mean', 'planet_y_std',
        'planet_z_mean', 'planet_z_std',
        'parallel_x_mean', 'parallel_x_std',
        'parallel_y_mean', 'parallel_y_std',
        'parallel_z_mean', 'parallel_z_std'
    ]
    fault_df = pd.DataFrame(features, columns=feature_cols)
    fault_df['label'] = labels
    fault_df['load_config'] = load_configs  # 可选：添加负载配置列

    # 均衡采样（每类1000条）
    balanced_df = pd.DataFrame()
    n_classes = 10  # 轴承5类+齿轮5类
    for label in range(n_classes):
        cls_df = fault_df[fault_df['label'] == label]
        if len(cls_df) == 0:
            print(f"警告：标签 {label} 无样本，跳过")
            continue
        sample_num = 1000 if len(cls_df) >= 1000 else len(cls_df)
        cls_df = cls_df.sample(n=sample_num, replace=(len(cls_df) < 1000), random_state=42)
        balanced_df = pd.concat([balanced_df, cls_df], ignore_index=True)

    print(f"特征提取完成！生成 {len(balanced_df)} 条样本")
    return balanced_df

# ---------------------- 3. 主函数：运行模型（修正根路径，添加异常处理） ----------------------
if __name__ == "__main__":
    # ------------ （1）配置数据集路径（改为你的实际路径，用r前缀避免转义） ------------
    root_path = r"D:/大文件/文献/齿轮箱故障诊断/读取数据集"
     # 启动前检查根目录
    if not os.path.exists(root_path):
        print(f"致命错误：根目录不存在 → {root_path}")
        print("请检查路径是否正确，或修改main.py中的root_path变量")
        exit(1)
    if not os.path.isdir(root_path):
        print(f"致命错误：{root_path} 不是一个有效的目录")
        exit(1)
    if not os.access(root_path, os.R_OK):
        print(f"致命错误：没有权限访问目录 → {root_path}")
        print("请检查文件夹权限设置")
        exit(1)
    
    # 读取数据
    try:
        csv_acc_data = load_seu_gearbox_data(root_path)
    except Exception as e:
        print(f"数据加载失败：{str(e)}")
        exit(1)

    # ------------ （2）读取数据并绘制时域图 ------------
    plot_time_domain_signal(csv_acc_data)
    import signal

    # ------------ （3）提取特征，生成模型输入数据 ------------
    fault_df = extract_features(csv_acc_data)
    # 修复：判断特征提取是否成功
    if fault_df.empty:
        print("错误：特征提取失败，终止模型运行！")
    else:
        # ------------ （4）初始化模型并运行案例1 ------------
        xai = XAI_WindTurbine(n_clusters=2, n_shap_samples=10000)
        print("\n" + "="*50)
        print("案例1：风机齿轮箱健康状态监测")
        print("="*50)
        health_res, shap_health = xai.health_monitoring(fault_df)

        # ------------ （5）保存结果 ------------
        health_res.to_excel("case1_health_monitoring_results.xlsx", index=False)
        print("\n结果已保存！")
        print("\n" + "="*50)
# ------------ （6）运行案例2：故障定位 ------------
print("\n" + "="*50)
print("案例2：风机齿轮箱故障定位")
print("="*50)

# 加载SCADA数据
scada_data_path = r"D:/大文件/文献/齿轮箱故障诊断/读取数据集/scada/测试.CSV"    # 替换为真实路径
try:
    scada_df = pd.read_csv(scada_data_path)
    print(f"成功加载SCADA数据，共 {len(scada_df)} 条记录")
except FileNotFoundError:
    print(f"错误：SCADA数据文件不存在 → {scada_data_path}")
    exit(1)
except Exception as e:
    print(f"加载SCADA数据失败：{str(e)}")
    exit(1)

# 调用故障定位方法
fault_contribution, shap_fault = xai.fault_localization(scada_df)
fault_contribution.to_excel("case2_fault_localization_results.xlsx", index=False)