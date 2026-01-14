import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
import seaborn as sns

# 导入自定义模块
from utils import load_dataset  # 改为load_dataset函数
from msissa import MSISSA
from comparison_algorithms import PSO, SSA, DBO

# 数据加载与预处理
root_path = r"D:/大文件/文献/齿轮箱故障诊断/读取数据集" 
X_train, X_test, y_train, y_test, label_map = load_dataset(
    root_path=root_path,
    window_size=2048,
    step=1000,
    samples_per_class=200
)

# 打印数据信息
print(f"数据加载完成：")
print(f"训练集：{X_train.shape}, 测试集：{X_test.shape}")
print(f"标签映射：{label_map}")
print(f"各类别样本数（训练集）：{np.bincount(y_train)}")
print(f"各类别样本数（测试集）：{np.bincount(y_test)}")

# 划分验证集（用于优化算法的适应度计算）
from sklearn.model_selection import train_test_split
X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
    X_train, y_train, train_size=0.8, random_state=42, stratify=y_train
)

# ------------------------------2. 超参数优化（各算法）------------------------------
bounds = [(0.01, 100), (0.001, 10)]
pop_size = 50
max_iter = 200

# 2.1 MSISSA优化SVM
print("\n=== MSISSA优化SVM ===")
msissa = MSISSA(pop_size=pop_size, max_iter=max_iter, bounds=bounds)
msissa_best_params, msissa_best_acc = msissa.optimize(X_train_opt, y_train_opt, X_val_opt, y_val_opt)
print(f"MSISSA最优参数：C={msissa_best_params[0]:.2f}, gamma={msissa_best_params[1]:.3f}")

# 2.2 PSO优化SVM
print("\n=== PSO优化SVM ===")
pso = PSO(pop_size=pop_size, max_iter=max_iter, bounds=bounds)
pso_best_params, pso_best_acc = pso.optimize(X_train_opt, y_train_opt, X_val_opt, y_val_opt)
print(f"PSO最优参数：C={pso_best_params[0]:.2f}, gamma={pso_best_params[1]:.3f}")

# 2.3 SSA优化SVM
print("\n=== SSA优化SVM ===")
ssa = SSA(pop_size=pop_size, max_iter=max_iter, bounds=bounds)
ssa_best_params, ssa_best_acc = ssa.optimize(X_train_opt, y_train_opt, X_val_opt, y_val_opt)
print(f"SSA最优参数：C={ssa_best_params[0]:.2f}, gamma={ssa_best_params[1]:.3f}")

# 2.4 DBO优化SVM
print("\n=== DBO优化SVM ===")
dbo = DBO(pop_size=pop_size, max_iter=max_iter, bounds=bounds)
dbo_best_params, dbo_best_acc = dbo.optimize(X_train_opt, y_train_opt, X_val_opt, y_val_opt)
print(f"DBO最优参数：C={dbo_best_params[0]:.2f}, gamma={dbo_best_params[1]:.3f}")

# ------------------------------3. 模型训练与测试------------------------------
def train_test_svm(best_params, X_train, y_train, X_test, y_test, model_name):
    C, gamma = best_params
    svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{model_name}测试准确率：{acc:.4f}")
    return acc, cm, y_pred

# 训练所有模型
msissa_acc, msissa_cm, msissa_pred = train_test_svm(msissa_best_params, X_train, y_train, X_test, y_test, "MSISSA-SVM")
pso_acc, pso_cm, pso_pred = train_test_svm(pso_best_params, X_train, y_train, X_test, y_test, "PSO-SVM")
ssa_acc, ssa_cm, ssa_pred = train_test_svm(ssa_best_params, X_train, y_train, X_test, y_test, "SSA-SVM")
dbo_acc, dbo_cm, dbo_pred = train_test_svm(dbo_best_params, X_train, y_train, X_test, y_test, "DBO-SVM")

# ------------------------------4. 结果可视化（适配10类标签）------------------------------
# 4.1 准确率对比柱状图
models = ["PSO-SVM", "DBO-SVM", "SSA-SVM", "MSISSA-SVM"]
accs = [pso_acc, dbo_acc, ssa_acc, msissa_acc]
plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accs, palette="viridis")
plt.title("各模型故障诊断准确率对比（轴承+齿轮箱）", fontsize=14)
plt.ylabel("准确率", fontsize=12)
plt.ylim(0.9, 1.0)
for i, acc in enumerate(accs):
    plt.text(i, acc+0.001, f"{acc:.4f}", ha="center", fontsize=11)
plt.savefig("accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# 4.2 MSISSA-SVM混淆矩阵热图（适配10类标签）
# 生成标签名称列表（对应label_map）
label_names = [
    "轴承正常", "轴承滚子", "轴承内圈", "轴承外圈", "轴承复合",
    "齿轮正常", "齿轮缺损", "齿轮断齿", "齿轮根裂", "齿轮磨损"
]
plt.figure(figsize=(12, 10))
sns.heatmap(msissa_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.title("MSISSA-SVM混淆矩阵（10类故障）", fontsize=14)
plt.xlabel("预测标签", fontsize=12)
plt.ylabel("真实标签", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.savefig("msissa_svm_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# ------------------------------5. 结果总结------------------------------
print("\n=== 实验结果总结 ===")
for model, acc in zip(models, accs):
    print(f"{model}: {acc:.4f}")
print(f"\nMSISSA-SVM相较于其他模型，准确率提升：")
print(f"- 比PSO-SVM提升：{msissa_acc - pso_acc:.4f}")
print(f"- 比DBO-SVM提升：{msissa_acc - dbo_acc:.4f}")
print(f"- 比SSA-SVM提升：{msissa_acc - ssa_acc:.4f}")