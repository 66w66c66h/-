import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 粒子群优化（PSO）
class PSO:
    def __init__(self, pop_size=50, max_iter=200, dim=2, bounds=[(0.01, 100), (0.001, 10)], 
                 w=0.5, c1=2, c2=2):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = dim
        self.bounds = np.array(bounds)
        self.w = w  # 惯性权重
        self.c1 = c1  # 认知因子
        self.c2 = c2  # 社会因子
        self.pop = None
        self.v = None  # 速度
        self.fitness = None
        self.best_params = None
        self.best_fitness = -np.inf

    def init_pop(self):
        self.pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.pop_size, self.dim))
        self.v = np.random.uniform(-1, 1, (self.pop_size, self.dim))

    def fitness_function(self, params, X_train, y_train, X_val, y_val):
        C, gamma = params
        svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        svm.fit(X_train, y_train)
        return accuracy_score(y_val, svm.predict(X_val))

    def optimize(self, X_train, y_train, X_val, y_val):
        self.init_pop()
        self.fitness = np.array([self.fitness_function(ind, X_train, y_train, X_val, y_val) 
                                 for ind in self.pop])
        p_best = self.pop.copy()  # 个体最优
        p_best_fitness = self.fitness.copy()
        self.best_idx = np.argmax(self.fitness)
        self.best_params = self.pop[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]

        for t in range(self.max_iter):
            # 更新速度和位置
            r1, r2 = np.random.random((2, self.pop_size, self.dim))
            self.v = self.w * self.v + self.c1 * r1 * (p_best - self.pop) + self.c2 * r2 * (self.best_params - self.pop)
            self.pop = self.pop + self.v
            self.pop = np.clip(self.pop, self.bounds[:, 0], self.bounds[:, 1])

            # 更新适应度
            new_fitness = np.array([self.fitness_function(ind, X_train, y_train, X_val, y_val) 
                                    for ind in self.pop])
            # 更新个体最优
            mask = new_fitness > p_best_fitness
            p_best[mask] = self.pop[mask]
            p_best_fitness[mask] = new_fitness[mask]
            # 更新全局最优
            current_best_idx = np.argmax(new_fitness)
            if new_fitness[current_best_idx] > self.best_fitness:
                self.best_fitness = new_fitness[current_best_idx]
                self.best_params = self.pop[current_best_idx].copy()
        return self.best_params, self.best_fitness

#原始麻雀算法（SSA）
class SSA:
    def __init__(self, pop_size=50, max_iter=200, dim=2, bounds=[(0.01, 100), (0.001, 10)], 
                 p_discoverer=0.2, p_alert=0.2, ST=0.8):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = dim
        self.bounds = np.array(bounds)
        self.p_discoverer = p_discoverer
        self.p_alert = p_alert
        self.ST = ST
        self.pop = None
        self.fitness = None
        self.best_params = None
        self.best_fitness = -np.inf

    def init_pop(self):
        self.pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.pop_size, self.dim))

    def fitness_function(self, params, X_train, y_train, X_val, y_val):
        C, gamma = params
        svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        svm.fit(X_train, y_train)
        return accuracy_score(y_val, svm.predict(X_val))

    def optimize(self, X_train, y_train, X_val, y_val):
        self.init_pop()
        n_discoverer = int(self.pop_size * self.p_discoverer)
        n_alert = int(self.pop_size * self.p_alert)

        self.fitness = np.array([self.fitness_function(ind, X_train, y_train, X_val, y_val) 
                                 for ind in self.pop])
        self.best_idx = np.argmax(self.fitness)
        self.best_params = self.pop[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]
        self.worst_params = self.pop[np.argmin(self.fitness)].copy()

        for t in range(self.max_iter):
            # 更新发现者
            R2 = np.random.random()
            A = np.random.choice([-1, 1], size=self.dim)
            Q = np.random.normal(0, 1)
            a = np.random.uniform(0, 1)
            for i in range(n_discoverer):
                if R2 < self.ST:
                    self.pop[i] = self.pop[i] * np.exp(-i/(a * self.max_iter))
                else:
                    self.pop[i] = self.pop[i] + Q * A
                self.pop[i] = np.clip(self.pop[i], self.bounds[:, 0], self.bounds[:, 1])

            # 更新警戒者
            for i in range(n_discoverer, n_discoverer + n_alert):
                beta = np.random.normal(0, 1)
                K = np.random.uniform(-1, 1)
                fi = self.fitness_function(self.pop[i], X_train, y_train, X_val, y_val)
                fb = self.best_fitness
                fw = np.min(self.fitness)
                if fi > fb:
                    self.pop[i] = self.best_params + beta * np.abs(self.pop[i] - self.best_params)
                else:
                    self.pop[i] = self.pop[i] + K * (np.abs(self.pop[i] - self.worst_params) / (fi - fw + 1e-10))
                self.pop[i] = np.clip(self.pop[i], self.bounds[:, 0], self.bounds[:, 1])

            # 更新加入者
            for i in range(n_discoverer + n_alert, self.pop_size):
                if i > self.pop_size / 2:
                    self.pop[i] = Q * np.exp((self.worst_params - self.pop[i])/i**2)
                else:
                    A_plus = A.reshape(-1, 1) @ np.linalg.pinv(A.reshape(1, -1) @ A.reshape(-1, 1)) @ A.reshape(1, -1)
                    self.pop[i] = self.best_params + np.abs(self.pop[i] - self.best_params) * A_plus
                self.pop[i] = np.clip(self.pop[i], self.bounds[:, 0], self.bounds[:, 1])

            # 更新适应度和最优解
            new_fitness = np.array([self.fitness_function(ind, X_train, y_train, X_val, y_val) 
                                    for ind in self.pop])
            current_best_idx = np.argmax(new_fitness)
            if new_fitness[current_best_idx] > self.best_fitness:
                self.best_fitness = new_fitness[current_best_idx]
                self.best_params = self.pop[current_best_idx].copy()
            self.worst_params = self.pop[np.argmin(new_fitness)].copy()
        return self.best_params, self.best_fitness

# 果蝇优化算法（DBO）
class DBO:
    def __init__(self, pop_size=50, max_iter=200, dim=2, bounds=[(0.01, 100), (0.001, 10)], 
                 smell_radius=10, vision_radius=5):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = dim
        self.bounds = np.array(bounds)
        self.smell_radius = smell_radius  # 嗅觉半径
        self.vision_radius = vision_radius  # 视觉半径
        self.pop = None
        self.fitness = None
        self.best_params = None
        self.best_fitness = -np.inf

    def init_pop(self):
        self.pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.pop_size, self.dim))

    def fitness_function(self, params, X_train, y_train, X_val, y_val):
        C, gamma = params
        svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        svm.fit(X_train, y_train)
        return accuracy_score(y_val, svm.predict(X_val))

    def optimize(self, X_train, y_train, X_val, y_val):
        self.init_pop()
        self.fitness = np.array([self.fitness_function(ind, X_train, y_train, X_val, y_val) 
                                 for ind in self.pop])
        self.best_idx = np.argmax(self.fitness)
        self.best_params = self.pop[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]

        for t in range(self.max_iter):
            # 果蝇搜索（嗅觉阶段）
            for i in range(self.pop_size):
                # 随机飞行
                self.pop[i] = self.pop[i] + np.random.uniform(-self.smell_radius, self.smell_radius, self.dim)
                self.pop[i] = np.clip(self.pop[i], self.bounds[:, 0], self.bounds[:, 1])

            # 视觉阶段（向最优个体靠近）
            distances = np.linalg.norm(self.pop - self.best_params, axis=1)
            for i in range(self.pop_size):
                if distances[i] < self.vision_radius:
                    self.pop[i] = self.pop[i] + np.random.uniform(0, 1) * (self.best_params - self.pop[i])
                self.pop[i] = np.clip(self.pop[i], self.bounds[:, 0], self.bounds[:, 1])

            # 更新适应度和最优解
            new_fitness = np.array([self.fitness_function(ind, X_train, y_train, X_val, y_val) 
                                    for ind in self.pop])
            current_best_idx = np.argmax(new_fitness)
            if new_fitness[current_best_idx] > self.best_fitness:
                self.best_fitness = new_fitness[current_best_idx]
                self.best_params = self.pop[current_best_idx].copy()
        return self.best_params, self.best_fitness