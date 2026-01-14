import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score as NMI
from data_preprocessor import DataPreprocessor
from shapley_calculator import ShapleyCalculator

class XAI_WindTurbine:
    def __init__(self, n_clusters=10, n_shap_samples=10000):
        self.n_clusters = n_clusters  # 案例1：5类齿轮箱状态（论文2.1节）
        self.n_shap_samples = n_shap_samples  # Shapley采样次数
        self.preprocessor = DataPreprocessor()
    
    def _build_clustering_models(self):
        """构建聚类模型（论文表1参数：KMeans/GMM/Spectral）"""
        return {
            'KMeans': KMeans(
                n_clusters=self.n_clusters,
                init='k-means++',
                n_init=10,
                max_iter=1000,
                random_state=42
            ),
            'GMM': GaussianMixture(
                n_components=self.n_clusters,
                init_params='kmeans',
                n_init=10,
                max_iter=1000,
                random_state=42
            ),
            'Spectral': SpectralClustering(
                n_clusters=self.n_clusters,
                n_init=10,
                random_state=42
            )
        }
    
    def _build_classification_model(self):
        """构建Xgboost分类模型（论文表2参数：案例1健康状态分类）"""
        return xgb.XGBClassifier(
            booster='gbtree',
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            objective='multi:softmax',
            num_class=self.n_clusters,
            random_state=42
        )
    
    def _build_regression_models(self):
        """构建回归模型（论文表5参数：案例2轴承温度预测）"""
        return {
            'Xgboost': xgb.XGBRegressor(
                booster='gbtree',
                n_estimators=1000,
                learning_rate=0.1,
                max_depth=4,
                min_child_weight=100,
                objective='reg:squarederror',
                random_state=42
            ),
            'CatBoost': cb.CatBoostRegressor(
                boosting_type='Plain',
                n_estimators=1000,
                learning_rate=0.1,
                max_depth=4,
                reg_lambda=3,
                random_state=42,
                verbose=0
            ),
            'LightGBM': lgb.LGBMRegressor(
                boosting_type='gbdt',
                n_estimators=1000,
                learning_rate=0.1,
                max_depth=4,
                min_child_weight=100,
                random_state=42,
                verbose=0
            )
        }
    
    def health_monitoring(self, df):
        # 1. 数据预处理（论文2.1节）
        df, feature_cols = self.preprocessor.prepare_fault_dataset(df)
        X = df[feature_cols].values
        y_true = df['label'].values
        
        # 2. 数据集划分（论文2.2节 2:8拆分训练集S/测试集T）
        train_size = int(0.2 * len(df))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y_true[:train_size], y_true[train_size:]
        
        # 3. 无监督聚类（R1 原始特征聚类 论文2.3.1节）
        clustering_models = self._build_clustering_models()
        cluster_results = {}
        for name, model in clustering_models.items():
            y_pred = model.fit_predict(X_test)
            cluster_results[name] = y_pred
        
        # 4. 有监督分类（R2 Xgboost预测 论文2.3.2节）
        clf_model = self._build_classification_model()
        clf_model.fit(X_train, y_train)
        y_pred_clf = clf_model.predict(X_test)
        
        # 5. Shapley值聚类（基于Shapley值优化聚类 论文2.3.3节）
        shap_calc = ShapleyCalculator(clf_model, self.n_shap_samples)
        shap_values = shap_calc.calculate_shap_values(X_test, feature_cols)
        
        shap_cluster_results = {}
        for name, model in clustering_models.items():
            y_pred_shap = model.fit_predict(shap_values)
            shap_cluster_results[name] = y_pred_shap
        
        # 6. 结果评估（NMI指标，论文表3）
        results = []
        for name in clustering_models.keys():
            results.append({
                'model': name,
                'nmi_raw_feature': round(NMI(y_test, cluster_results[name]), 4),
                'nmi_classification': round(NMI(y_test, y_pred_clf), 4),
                'nmi_shap_feature': round(NMI(y_test, shap_cluster_results[name]), 4)
            })
        
        results_df = pd.DataFrame(results)
        print("案例1：健康状态监测NMI结果（参考论文表3）：")
        print(results_df)
        shap_calc.plot_feature_importance(shap_values)  # 绘制特征重要性
        return results_df, shap_values
    
    def fault_localization(self, df):
        """案例2：风机齿轮箱故障定位（论文3节：轴承温度异常定位）"""
        # 1. 数据预处理（论文3.1节：SCADA数据清洗）
        df, feature_cols = self.preprocessor.clean_scada_data(df)
        X = df[feature_cols].values
        y = df['bearing_temperature'].values
        
        # 2. 数据集划分（论文3.2节：1-6月训练，7月测试）
        train_mask = df.index < int(0.85 * len(df))  # 近似1-6月数据占比
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]
        
        # 3. 训练回归模型（论文表5/6：选择MAE最小的Xgboost）
        reg_models = self._build_regression_models()
        best_model, best_mae = None, float('inf')
        for name, model in reg_models.items():
            model.fit(X_train, y_train)
            mae = np.mean(np.abs(model.predict(X_test) - y_test))
            print(f"{name} 回归模型MAE：{mae:.4f}")
            if mae < best_mae:
                best_mae = mae
                best_model = model
        
        # 4. 异常检测（论文3.3.2节：3倍标准差法）
        mu, sigma = np.mean(y_test), np.std(y_test)
        anomaly_threshold = mu + 3 * sigma
        anomaly_mask = y_test > anomaly_threshold
        X_anomaly = X_test[anomaly_mask]
        print(f"\n检测到异常样本数：{sum(anomaly_mask)}（阈值：{anomaly_threshold:.2f}）")
        
        # 5. Shapley值故障定位（论文3.3.4节）
        shap_calc = ShapleyCalculator(best_model, self.n_shap_samples)
        shap_values = shap_calc.calculate_shap_values(X_anomaly, feature_cols)
        
        # 特征贡献度排序（匹配论文图8）
        feature_contribution = pd.DataFrame({
            'feature': feature_cols,
            'avg_shap_value': np.mean(shap_values, axis=0)
        }).sort_values('avg_shap_value', ascending=False)
        
        print("\n案例2：故障特征贡献度排序（参考论文图8）：")
        print(feature_contribution)
        
        # 定位主要故障原因（论文3.3.4节：F6油温异常）
        main_fault = feature_contribution.iloc[0]['feature']
        scada_map = {
            'F6': '齿轮箱入口油温', 'F9': '发电机扭矩', 'F5': '齿轮箱转速', 'F1': '叶轮转速'
        }
        print(f"\n主要故障原因：{main_fault}（对应SCADA项：{scada_map.get(main_fault, main_fault)}）")
        return feature_contribution, shap_values