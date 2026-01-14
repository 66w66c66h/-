import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class MSISSA:
    def __init__(self, pop_size=50, max_iter=200, dim=2, bounds=[(0.01, 100), (0.001, 10)], 
                 p_discoverer=0.2, p_alert=0.2, ST=0.8, k=2):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = dim
        self.bounds = np.array(bounds)
        self.p_discoverer = p_discoverer
        self.p_alert = p_alert
        self.ST = ST
        self.k = k
        self.pop = None  # 种群位置
        self.fitness = None  # 适应度值
        self.best_fitness = -np.inf  # 最优适应度
        self.best_params = None  # 最优超参数（C, gamma）

    def latin_hypercube_init(self):  #拉丁超立方初始化
        pop = np.zeros((self.pop_size, self.dim))
        for d in range(self.dim):
            # 划分区间并生成随机排列
            intervals = np.linspace(self.bounds[d, 0], self.bounds[d, 1], self.pop_size+1)
            perm = np.random.permutation(self.pop_size)
            # 每个区间随机抽样
            for i in range(self.pop_size):
                pop[i, d] = np.random.uniform(intervals[perm[i]], intervals[perm[i]+1])
        return pop

    def adaptive_weight(self, t):
        # 自适应权重因子
        return 0.2 * np.cos(np.pi/2 * (1 - t/self.max_iter))

    def levy_flight(self, alpha=1.5):
        # 莱维飞行步长（Mantegna算法）
        sigma_m = (np.math.gamma(1+alpha) * np.sin(np.pi*alpha/2) / 
                   (np.math.gamma((1+alpha)/2) * alpha * 2**((alpha-1)/2))) ** (1/alpha)
        sigma_n = 1
        m = np.random.normal(0, sigma_m**2)
        n = np.random.normal(0, sigma_n**2)
        s = m / (np.abs(n)**(1/alpha))
        return 0.01 * s  # 步长控制（论文公式20）

    def variable_spiral(self, i, npop):
        # 可变螺旋搜索因子
        l = np.random.uniform(-1, 1)
        z = np.exp(self.k * np.cos(np.pi * (npop - i)/npop))
        phi = z * np.exp(l) * np.cos(2 * np.pi * l)
        return phi

    def fitness_function(self, params, X_train, y_train, X_val, y_val):
        # 适应度函数：SVM分类准确率
        C, gamma = params
        svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_val)
        return accuracy_score(y_val, y_pred)

    def optimize(self, X_train, y_train, X_val, y_val):
        # 1. 初始化种群（拉丁超立方）
        self.pop = self.latin_hypercube_init()
        n_discoverer = int(self.pop_size * self.p_discoverer)
        n_alert = int(self.pop_size * self.p_alert)

        # 2. 初始适应度计算
        self.fitness = np.array([self.fitness_function(ind, X_train, y_train, X_val, y_val) 
                                 for ind in self.pop])
        self.best_idx = np.argmax(self.fitness)
        self.best_fitness = self.fitness[self.best_idx]
        self.best_params = self.pop[self.best_idx].copy()
        self.worst_fitness = np.min(self.fitness)
        self.worst_params = self.pop[np.argmin(self.fitness)].copy()

        #  迭代优化
        for t in range(self.max_iter):
            # 3.1 更新发现者位置（改进策略2）
            omega = self.adaptive_weight(t)
            R2 = np.random.random()  # 预警值
            A = np.random.choice([-1, 1], size=self.dim) 
            Q = np.random.normal(0, 1) 
            for i in range(n_discoverer):
                if R2 < self.ST:
                    self.pop[i] = omega * self.pop[i] * np.exp(-1/(omega - self.pop_size))
                else:
                    self.pop[i] = omega * self.pop[i] + Q * A
                # 边界约束
                self.pop[i] = np.clip(self.pop[i], self.bounds[:, 0], self.bounds[:, 1])

            # 3.2 更新警戒者位置（改进策略3）
            for i in range(n_discoverer, n_discoverer + n_alert):
                levy_step = self.levy_flight()
                self.pop[i] = self.pop[i] + levy_step * (self.best_params - self.pop[i])
                self.pop[i] = np.clip(self.pop[i], self.bounds[:, 0], self.bounds[:, 1])

            # 3.3 更新加入者位置（改进策略4）
            for i in range(n_discoverer + n_alert, self.pop_size):
                phi = self.variable_spiral(i, self.pop_size)
                if i > self.pop_size / 2:
                    # 适应度差，随机迁移
                    self.pop[i] = phi * Q * np.exp((self.worst_params - self.pop[i])/i**2)
                else:
                    # 靠近发现者最优位置
                    A_plus = A.reshape(-1, 1) @ np.linalg.pinv(A.reshape(1, -1) @ A.reshape(-1, 1)) @ A.reshape(1, -1)
                    self.pop[i] = self.best_params + np.abs(self.pop[i] - self.best_params) * A_plus * phi
                self.pop[i] = np.clip(self.pop[i], self.bounds[:, 0], self.bounds[:, 1])

            # 3.4 更新适应度和最优解
            new_fitness = np.array([self.fitness_function(ind, X_train, y_train, X_val, y_val) 
                                    for ind in self.pop])
            # 保留更优个体
            mask = new_fitness > self.fitness
            self.fitness[mask] = new_fitness[mask]
            self.pop[mask] = self.pop[mask]

            # 更新全局最优
            current_best_idx = np.argmax(self.fitness)
            if self.fitness[current_best_idx] > self.best_fitness:
                self.best_fitness = self.fitness[current_best_idx]
                self.best_params = self.pop[current_best_idx].copy()
            self.worst_fitness = np.min(self.fitness)
            self.worst_params = self.pop[np.argmin(self.fitness)].copy()

            # 打印迭代信息
            if (t+1) % 50 == 0:
                print(f"迭代{t+1}/{self.max_iter} | 最优适应度：{self.best_fitness:.4f} | 最优参数：C={self.best_params[0]:.2f}, gamma={self.best_params[1]:.3f}")

        return self.best_params, self.best_fitness