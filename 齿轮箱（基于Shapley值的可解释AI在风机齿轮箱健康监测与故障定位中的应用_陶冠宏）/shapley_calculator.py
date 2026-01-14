import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import shap

class ShapleyCalculator:
    def __init__(self, model, n_samples=10000):
        self.model = model
        self.n_samples = n_samples
        self.feature_names = None
        self.explainer = shap.TreeExplainer(model)
        self.feature_mean = None  # 存储特征均值，用于填充缺失特征
    
    def _marginal_contribution(self, X, feature_idx, permutation):
        """计算单个特征排列下的边际贡献（论文公式2）"""
        # 构造包含特征i的子集S∪{i}和不包含的子集S
        n_samples, n_features = X.shape
        idx = np.where(permutation == feature_idx)[0][0]
        S_with = permutation[:idx+1]  # 包含特征i的子集
        S_without = permutation[:idx]  # 不包含特征i的子集
        
        '''
        # 处理空子集情况
        X_with = X[:, S_with] if len(S_with) > 0 else np.zeros((X.shape[0], 1))
        X_without = X[:, S_without] if len(S_without) > 0 else np.zeros((X.shape[0], 1))
        
        # 适配Xgboost模型输入格式
        if isinstance(self.model, xgb.Booster):
            pred_with = self.model.predict(xgb.DMatrix(X_with))
            pred_without = self.model.predict(xgb.DMatrix(X_without))
        else:
            pred_with = self.model.predict(X_with)
            pred_without = self.model.predict(X_without)
        
        return pred_with - pred_without  # 边际贡献 = 包含特征i的预测值 - 不包含的预测值
        '''
        X_with = X.copy()
        # 对于不在S_with中的特征，用均值填充
        for f in range(n_features):
            if f not in S_with:
                X_with[:, f] = self.feature_mean[f]
    
        X_without = X.copy()
        # 对于不在S_without中的特征，用均值填充（此时特征i一定不在其中）
        for f in range(n_features):
            if f not in S_without:
                X_without[:, f] = self.feature_mean[f]
    
        # 适配模型输入格式（保持特征数量为12）
        if isinstance(self.model, xgb.Booster):
            pred_with = self.model.predict(xgb.DMatrix(X_with))
            pred_without = self.model.predict(xgb.DMatrix(X_without))
        else:
            pred_with = self.model.predict(X_with)
            pred_without = self.model.predict(X_without)
    
        return pred_with - pred_without
    
    def calculate_shap_values(self, X, feature_names):
        """计算Shapley值（论文公式3/4：蒙特卡洛平均边际贡献）"""
        self.feature_names = feature_names
        n_samples, n_features = X.shape
        self.feature_mean = np.mean(X, axis=0)  # 计算特征均值作为参考值
        shap_values = np.zeros((n_samples, n_features))
        
        # 蒙特卡洛随机采样（论文1节：复杂度O(Km)，K=采样次数，m=特征数）
        for _ in tqdm(range(self.n_samples), desc="Calculating Shapley Values"):
            permutation = np.random.permutation(n_features)  # 随机特征排列
            for i in range(n_features):
                mc = self._marginal_contribution(X, i, permutation)
                shap_values[:, i] += mc
        
        # 无偏估计：平均边际贡献（论文公式3）
        shap_values /= self.n_samples
        return shap_values
    
    def plot_feature_importance(self, shap_values):
        """绘制特征重要性图（论文图8：按平均绝对Shapley值排序）"""
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': mean_abs_shap
        }).sort_values('shap_importance', ascending=True)
        
        # 绘制水平条形图（匹配论文图8风格）
        plt.figure(figsize=(10, 6))
        sns.barplot(x='shap_importance', y='feature', data=feature_importance)
        plt.title('Feature Importance (Mean Absolute Shapley Value)', fontsize=12)
        plt.xlabel('Mean Absolute Shapley Value', fontsize=10)
        plt.tight_layout()
        plt.savefig('shap_feature_importance.png', dpi=300)  # 保存结果图
        