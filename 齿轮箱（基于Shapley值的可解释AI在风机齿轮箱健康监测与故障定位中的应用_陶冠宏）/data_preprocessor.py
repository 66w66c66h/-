import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # 论文3.1节：最大最小标准化
    
    def clean_scada_data(self, df):
        """清洗SCADA数据（论文3.1节：异常处理、特征构造）"""
        # 1. 删除停机数据（有功功率≤0或风速<0，论文3.1节）
        df = df[df['active_power'] > 0]
        df = df[df['wind_speed'] > 0]
        
        # 2. 构造桨叶角度偏差特征F10（论文公式5：Dev = 1/3 * Σ|θi - μ|）
        if all(col in df.columns for col in ['blade1_angle', 'blade2_angle', 'blade3_angle']):
            blade_angles = df[['blade1_angle', 'blade2_angle', 'blade3_angle']].values
            mu = np.mean(blade_angles, axis=1, keepdims=True)
            df['F10'] = np.mean(np.abs(blade_angles - mu), axis=1)
        
        # 3. 选择论文表4的12个特征，保留轴承温度作为目标变量
        feature_cols = [
            'F1', 'F2', 'F3', 'F4', 'F5', 'F6',
            'F7', 'F8', 'F9', 'F10', 'F11', 'F12'
        ]
        df = df[feature_cols + ['bearing_temperature']].dropna()
        
        # 4. 标准化到[0,1]区间（论文3.1节）
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        return df, feature_cols
    
    def prepare_fault_dataset(self, df, n_classes=10):
        """准备案例1故障诊断数据集（论文2.1节：5类状态，每类4万条数据）"""
        # 选择前3个有效字段：X轴加速度、Y轴加速度、转速（论文2.1节）
        feature_cols = [
            'planet_x_mean', 'planet_x_std',
            'planet_y_mean', 'planet_y_std',
            'planet_z_mean', 'planet_z_std',
            'parallel_x_mean', 'parallel_x_std',
            'parallel_y_mean', 'parallel_y_std',
            'parallel_z_mean', 'parallel_z_std'
            ]
        df = df[feature_cols + ['label']].dropna()
        
        # 标准化
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        
        # 按类别均衡采样（每类4万条，论文2.1节）
        balanced_df = pd.DataFrame()
        #替换原本40000条数据为1000条
        for label in range(n_classes):
            cls_df = df[df['label'] == label]
            # 修改：将40000改为1000
            if len(cls_df) < 1000:
               cls_df = cls_df.sample(n=1000, replace=True, random_state=42)
            else:
               cls_df = cls_df.sample(n=1000, random_state=42)
            balanced_df = pd.concat([balanced_df, cls_df], ignore_index=True)

        return balanced_df, feature_cols