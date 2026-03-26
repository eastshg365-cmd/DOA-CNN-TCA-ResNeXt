# TCA+MUSIC+CNN DOA 估计算法：完整数学推导

---

## 一、TCA 阵列几何

### 1.1 参数定义

| 符号 | 数值 | 定义 |
|------|------|------|
| M | 5 | 第一子阵倍数（奇数） |
| Narr | 6 | 第二子阵倍数 |
| d | λ/2 | 传感器间距（半波长），归一化后 d=1 |
| P | 12 | 总传感器数 |
| D | 54d | 阵列孔径（首末传感器距离） |

### 1.2 互质条件

$$\gcd(M, \text{Narr}) = \gcd(5, 6) = 1$$

互质保证差分阵列无穿孔（connected co-array），是 TCA 有效的充要条件。

### 1.3 三子阵构造

$$X_1 = \{nM : n = 0,1,\ldots,\text{Narr}-1\} = \{0, 5, 10, 15, 20, 25\}$$

$$X_2 = \{m\cdot\text{Narr} : m = 1,\ldots,\lfloor M/2\rfloor\} = \{6, 12\}$$

$$X_3 = \{(m+M+1)\cdot\text{Narr} : m = 0,\ldots,M-2\} = \{36, 42, 48, 54\}$$

**设计原理**：
- X₁ 是以 M 为步长的均匀子阵（Narr 个元素）
- X₂ 是以 Narr 为步长的短子阵（填充低孔径间隙）
- X₃ 是高孔径扩展子阵（拉伸虚拟孔径）

$$\text{TCA} = \text{sort}(X_1 \cup X_2 \cup X_3) = \{0,5,6,10,12,15,20,25,36,42,48,54\}$$

### 1.4 自由度（DOF）分析

**ULA（均匀线阵）DOF**：P - 1 = 11（只能估计 ≤11 个信源）

**TCA 差分孔径（Virtual ULA）DOF**：

差分阵列 $\mathcal{D} = \{p_i - p_j : p_i, p_j \in \text{TCA}\}$

连续差分段 $\mathcal{L} = \{0,1,2,\ldots,L_{\max}\}$，DOF = 2L_max + 1

TCA 理论上 DOF ≈ 2MN - 1 = 59，本例实测 = **69**（超过理论下界）

> **物理意义**：12 个物理传感器产生等效 69 个虚拟传感器，可估计最多 34 个信源。

---

## 二、信号模型

### 2.1 接收信号模型

$$\mathbf{x}(t) = \mathbf{A}(\boldsymbol{\theta})\mathbf{s}(t) + \mathbf{n}(t), \quad t = 1,\ldots,T$$

| 符号 | 维度 | 定义 |
|------|------|------|
| $\mathbf{x}(t)$ | P×1 | 接收信号向量（复数） |
| $\mathbf{A}(\boldsymbol{\theta})$ | P×K | 导向矩阵 |
| $\mathbf{s}(t)$ | K×1 | 信源信号向量 |
| $\mathbf{n}(t)$ | P×1 | 加性高斯白噪声 |
| T | 16 | 快拍数（观测次数） |
| K | 3 | 信源数 |

### 2.2 导向向量

$$\mathbf{a}(\theta) = \left[e^{j\pi p_1 \sin\theta},\ e^{j\pi p_2 \sin\theta},\ \ldots,\ e^{j\pi p_P \sin\theta}\right]^T \in \mathbb{C}^{P}$$

其中 $p_i$ 是 TCA 第 i 个传感器位置（以 d=λ/2 为单位）。

**验证 θ=0°**：$\sin(0) = 0$，所有元素 = $e^0 = 1$，模值全为 1 ✓

**验证 θ=30°**，第二元素（p=5）：
$$\arg\left(e^{j\pi \cdot 5 \cdot \sin(30°)}\right) = \pi \cdot 5 \cdot 0.5 = 2.5\pi \equiv 0.5\pi \pmod{2\pi} = 90°$$

### 2.3 噪声功率设定

$$\text{SNR} = \frac{\sigma_s^2}{\sigma_n^2} = 10^{\text{SNR}_{\text{dB}}/10} = 10^{10/10} = 10$$

$$\sigma_n = \sqrt{\frac{\sigma_s^2}{2 \cdot \text{SNR}}} \quad \text{（复数实虚部各贡献一半功率）}$$

---

## 三、协方差矩阵

### 3.1 理论协方差矩阵

$$\mathbf{R} = \mathbf{A}\mathbf{R}_s\mathbf{A}^H + \sigma_n^2\mathbf{I}_P \in \mathbb{C}^{P \times P}$$

- $\mathbf{R}_s = \mathbf{I}_K$（归一化，各信源独立且等功率）
- Hermitian 矩阵：$\mathbf{R} = \mathbf{R}^H$

### 3.2 样本协方差矩阵（MLE 估计）

$$\hat{\mathbf{R}} = \frac{1}{T}\sum_{t=1}^{T}\mathbf{x}(t)\mathbf{x}^H(t) = \frac{1}{T}\mathbf{X}\mathbf{X}^H$$

其中 $\mathbf{X} = [\mathbf{x}(1),\ldots,\mathbf{x}(T)] \in \mathbb{C}^{P \times T}$。

**当 T→∞ 时**，$\hat{\mathbf{R}} \to \mathbf{R}$（大数定律）。T=16 时存在统计波动。

---

## 四、MUSIC 算法

### 4.1 特征值分解

$$\hat{\mathbf{R}} = \mathbf{E}_s\boldsymbol{\Lambda}_s\mathbf{E}_s^H + \mathbf{E}_n\boldsymbol{\Lambda}_n\mathbf{E}_n^H$$

特征值降序排列：$\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_K \gg \lambda_{K+1} \approx \cdots \approx \lambda_P \approx \sigma_n^2$

| 子空间 | 特征向量矩阵 | 维度 | 含义 |
|--------|-------------|------|------|
| 信号子空间 $\mathcal{S}$ | $\mathbf{E}_s$ | P×K | 前 K 个最大特征向量 |
| 噪声子空间 $\mathcal{N}$ | $\mathbf{E}_n$ | P×(P-K) = 12×9 | 后 P-K 个最小特征向量 |

**关键性质**：$\text{span}(\mathbf{A}) = \text{span}(\mathbf{E}_s)$，即信号子空间与噪声子空间**正交**：

$$\mathbf{a}^H(\theta_k)\mathbf{E}_n = \mathbf{0}, \quad k = 1,\ldots,K$$

### 4.2 MUSIC 伪谱

$$P_{\text{MUSIC}}(\theta) = \frac{1}{\mathbf{a}^H(\theta)\mathbf{E}_n\mathbf{E}_n^H\mathbf{a}(\theta)}$$

- 当 $\theta = \theta_k$（真实 DOA）：分母 → 0，谱值 → +∞（尖峰）
- 其他方向：分母 > 0，谱值有限

**代码实现**：预计算投影矩阵 $\mathbf{U}_n = \mathbf{E}_n\mathbf{E}_n^H \in \mathbb{C}^{P \times P}$，避免重复计算。

### 4.3 仿真结果

$$\hat{\boldsymbol{\theta}}_{\text{MUSIC}} = \{-20°, 10°, 35°\}, \quad \text{True} = \{-20°, 10°, 35°\}$$

$$\text{误差} = \{3.6\times10^{-15}, 5.3\times10^{-15}, 7.1\times10^{-15}\}°$$

误差在机器精度（~10⁻¹⁵）量级，说明 SNR=10dB、T=16 下 MUSIC 精确收敛。

---

## 五、信源数估计：MDL/AIC（Wax-Kailath 1985）

### 5.1 问题设定

实际应用中 K 未知，需从数据估计。利用噪声子空间特征值"等于 σₙ²"这一特性。

### 5.2 关键统计量

设第 k 个噪声子空间假设下，噪声特征值为 $\{\lambda_{k+1}, \ldots, \lambda_P\}$：

**算术平均（AM）**：
$$\bar{\lambda}_a(k) = \frac{1}{P-k}\sum_{i=k+1}^{P}\lambda_i$$

**几何平均（GM）**：
$$\bar{\lambda}_g(k) = \left(\prod_{i=k+1}^{P}\lambda_i\right)^{1/(P-k)} = \exp\left(\frac{1}{P-k}\sum_{i=k+1}^P\log\lambda_i\right)$$

**AM-GM 不等式**：$\bar{\lambda}_a \geq \bar{\lambda}_g$，等号当且仅当所有特征值相等（真正是白噪声时成立）。

**对数似然比**（测量"多白"）：
$$\mathcal{L}(k) = \log\frac{\bar{\lambda}_g(k)}{\bar{\lambda}_a(k)} = \frac{1}{P-k}\sum_{i=k+1}^P\log\lambda_i - \log\bar{\lambda}_a(k) \leq 0$$

### 5.3 MDL 准则（复数高斯）

$$\text{MDL}(k) = -T \cdot \mathcal{L}(k) \cdot (P-k) + \frac{k(2P-k)}{2}\log T$$

展开为：

$$\boxed{\text{MDL}(k) = T\left[(P-k)\log\bar{\lambda}_a(k) - \sum_{i=k+1}^{P}\log\lambda_i\right] + \frac{k(2P-k)}{2}\log T}$$

| 项 | 含义 | 行为 |
|----|------|------|
| 第一项 | 负对数似然（数据拟合质量） | k↑ → 0（噪声越来越"白"） |
| 第二项 | 模型复杂度惩罚 | k↑ → 增大 |

**估计量**：$\hat{K} = \arg\min_{k=1,\ldots,P-1} \text{MDL}(k)$

### 5.4 AIC 准则

$$\text{AIC}(k) = 2T\left[(P-k)\log\bar{\lambda}_a(k) - \sum_{i=k+1}^{P}\log\lambda_i\right] + 2k(2P-k)$$

AIC 惩罚系数用 2，MDL 用 log(T)/2。T 大时 MDL 惩罚更重，更倾向于较小 K（一致性更好）。

### 5.5 自由参数数量推导

复数高斯模型参数个数（实数参数）：
- 信源协方差矩阵 $\mathbf{R}_s$（K×K 复数 Hermitian）：$K^2$ 个实参数
- 导向矩阵中未知的 K 个 DOA：K 个实参数
- 噪声功率 σₙ²：1 个实参数

模型阶数（去除冗余）：$p(k) = k^2 + k + 1 \approx k(2P-k)$ （Wax-Kailath 近似）

### 5.6 特征值比值法（小 T 鲁棒替代）

$$\hat{K} = \arg\max_{i=1,\ldots,P-1} \frac{\lambda_i}{\lambda_{i+1}}$$

**原理**：信号-噪声边界处特征值发生最大跳变。T=16 时仍有效（无需 T≫P 假设）。

---

## 六、CRB（Cramér-Rao Bound）理论下界

### 6.1 定义

CRB 给出任何无偏估计量方差的**下界**（理论最优性能）：

$$\text{Var}(\hat{\theta}_k) \geq \text{CRB}(\theta_k)$$

### 6.2 ULA 近似公式

对 P 元 ULA、T 快拍、SNR 线性值 ρ：

$$\text{CRB}(\theta_k) = \frac{6}{\rho \cdot T \cdot \pi^2 \cos^2(\theta_k)(P^3 - P)}$$

**RMSE 下界（度）**：

$$\text{RMSE}_{\min}(\theta_k) = \sqrt{\text{CRB}(\theta_k)} \cdot \frac{180}{\pi}$$

### 6.3 各参数影响

| 参数 | CRB ∝ | 物理含义 |
|------|--------|---------|
| T（快拍数） | 1/T | T 翻倍 → RMSE 降 √2 ≈ 3dB |
| P（传感器数） | 1/(P³-P) | P↑ 改善显著（立方律） |
| SNR | 1/ρ = 10^(-SNR/10) | SNR 增 10dB → RMSE 降 1/√10 |
| θ | 1/cos²θ | 边缘角 θ→90° 时精度急剧下降 |

**SNR=5dB 时 T 对比（P=12）**：

| T | RMSE 下界 |
|---|---------|
| 8 | 0.214° |
| 16 | 0.152° |
| 32 | 0.107° |
| 64 | 0.076° |

规律：T 翻倍 → RMSE × 1/√2（对数轴等间距平行线）。

---

## 七、BCE Loss 与 Focal Loss

### 7.1 Sigmoid 激活

$$\sigma(z) = \frac{1}{1+e^{-z}} \in (0,1)$$

### 7.2 Binary Cross-Entropy Loss（多标签）

对单个角度格 $j$，真实标签 $y_j \in \{0,1\}$，预测概率 $\hat{y}_j = \sigma(z_j)$：

$$\mathcal{L}_{\text{BCE}}(y_j, \hat{y}_j) = -y_j\log\hat{y}_j - (1-y_j)\log(1-\hat{y}_j)$$

**梯度推导**：

$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z_j} = \sigma(z_j) - y_j = \hat{y}_j - y_j$$

简洁优美，$\hat{y}_j > y_j$ 时梯度为正（需降低 z），$\hat{y}_j < y_j$ 时梯度为负（需提升 z）。

### 7.3 正负样本不均衡问题

本任务：121 个角度格中仅 K=3 个为正（信源存在）。

$$\text{正负比} = \frac{3}{121-3} = 1:39.3$$

问题：网络倾向于将所有格预测为0（负类），拟合简单负样本，忽略真正的信源位置。

### 7.4 Focal Loss（Lin et al. 2017）

$$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

其中 $p_t = \hat{y}$ (y=1) 或 $1-\hat{y}$ (y=0)，参数 γ=2，α=0.25。

**调制因子** $(1-p_t)^\gamma$：

| 预测置信度 | $(1-p_t)^2$ | 效果 |
|-----------|-------------|------|
| p=0.9（易分类） | $(0.1)^2 = 0.01$ | 损失 × 0.01，几乎忽略 |
| p=0.5（中等） | $(0.5)^2 = 0.25$ | 损失 × 0.25 |
| p=0.1（难分类） | $(0.9)^2 = 0.81$ | 损失保留 81%，聚焦难样本 |

**量化对比**（γ=2, α=0.25）：

| ŷ | BCE | Focal | 降低比例 |
|---|-----|-------|---------|
| 0.9 (easy) | 0.105 | 0.000263 | **99.75%** |
| 0.1 (hard) | 2.303 | 0.466 | 79.75% |

---

## 八、分类评估指标

### 8.1 混淆矩阵定义

$$\text{TP} = \sum_{i,j} y_{ij} \cdot \hat{y}_{ij}^{\text{bin}}, \quad \text{FP} = \sum_{i,j}(1-y_{ij})\cdot\hat{y}_{ij}^{\text{bin}}$$

$$\text{TN} = \sum_{i,j}(1-y_{ij})(1-\hat{y}_{ij}^{\text{bin}}), \quad \text{FN} = \sum_{i,j} y_{ij}(1-\hat{y}_{ij}^{\text{bin}})$$

### 8.2 各指标定义与含义

$$\text{Precision} = \frac{\text{TP}}{\text{TP}+\text{FP}} \quad \text{（预测为正中真正为正的比例）}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP}+\text{FN}} \quad \text{（真正为正中被正确预测的比例）}$$

$$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP}+\text{FP}+\text{FN}}$$

### 8.3 阈值优化

F1 对阈值 τ 的依赖：

$$\hat{y}^{\text{bin}} = \mathbb{1}[\hat{y} \geq \tau]$$

搜索 $\tau \in [0.1, 0.9]$：最优 τ = **0.75**，对应 F1 = **0.774**（vs 固定0.5时的0.492，提升57%）。

**物理原因**：正样本概率分布偏高（信源处 ŷ 接近1），提高阈值可有效减少 FP（虚报）。

---

## 九、差分阵列协方差向量化（欠定 DOA）

### 9.1 虚拟协方差向量

$$\mathbf{r}(d) = \frac{1}{|\mathcal{P}(d)|}\sum_{(i,j): p_i-p_j=d} [\hat{\mathbf{R}}]_{ij}, \quad d \in \mathcal{D}$$

等效于用 P² 个 $\hat{R}_{ij}$ 元素平均，生成虚拟均匀线阵的协方差向量。

### 9.2 虚拟传感器数量

$$|\mathcal{D}| = |\text{uniqueDiffs}| = 109 \quad \text{（本例）}$$

虚拟阵列具有 109 个虚拟传感器，远超物理 P=12 个。

---

---

## 十一、实验结论

### 11.1 TCA 阵列几何验证

M=5、Narr=6（gcd=1 互质满足）的 TCA 共有 **P=12 个物理传感器**、孔径 **54d**。差分阵列分析显示 TCA 可生成等效 **DOF=69 个虚拟传感器**，相比同等传感器数 ULA（DOF=11）提升 **527%**。

**结论**：12 个物理传感器产生 69 个虚拟传感器，最大可估计 34 个信源，为过定 DOA 估计提供充足自由度。

---

### 11.2 MUSIC 算法性能验证

SNR=10 dB、T=16、K=3（DOA={-20°, 10°, 35°}）条件下：

| 指标 | 结果 |
|------|------|
| MUSIC 估计 DOA | {-20.0°, 10.0°, 35.0°} |
| 估计误差 | ~10⁻¹⁵°（机器精度） |
| 噪声子空间维度 | 12×9（正确） |

误差在机器精度量级，TCA 的大虚拟孔径进一步增强子空间区分度。

**关键修复**：特征值降序排列中 `Ordering` 参数错误（`-1` 仅取最大索引）导致噪声子空间维度退化为 {1}，修正为 `Reverse[Ordering[...]]` 后恢复正常。

---

### 11.3 信源数估计对比

三种 K 估计方法在 K=3（T=16, SNR=10dB）条件下：

| 方法 | 估计结果 | 适用条件 | 复杂度 |
|------|---------|---------|--------|
| AIC（原始错误公式） | K=0（失败） | — | O(P²) |
| MDL（原始错误公式） | K=0（失败） | — | O(P²) |
| AIC（修正 Wax-Kailath） | **K=3 ✅** | T>P | O(P²) |
| MDL（修正 Wax-Kailath） | **K=3 ✅** | T>P | O(P²) |
| 特征值比值法 | **K=3 ✅** | T≥K+1 | O(P) |

**错误根源**：原公式对数似然项用信号特征值 $\sum_{i=1}^k\log\lambda_i$，应为**噪声特征值的 AM-GM 对数比** $\sum_{i=k+1}^P\log\lambda_i - (P-k)\log\bar{\lambda}_a$。修正后 T=16 即可正确收敛至 K=3。

**特征值比值法优势**：无需 T≫P 假设，2 行代码，T=16 小快拍条件下仍能准确识别 K=3，是工程实践中的首选方法。

---

### 11.4 评估指标优化效果

基于 100 个样本、121 类多标签分类（正负比 ≈1:40）的评估：

#### 阈值优化

| 配置 | Threshold | F1 Score | FP 数量 |
|------|-----------|----------|---------|
| **原始** | 0.50 | 0.492 | 586 |
| **优化后** | 0.75 | **0.774** | 显著降低 |
| 提升幅度 | +50% | **+57.3%** | — |

F1 vs 阈值曲线在 0.65~0.80 形成稳定平台，0.75 是工程鲁棒选择。**原因**：信源存在时预测概率 ŷ 接近 1，提高阈值可有效过滤低置信度虚报。

#### Focal Loss vs BCE 对比（γ=2, α=0.25）

| 预测置信度 | BCE Loss | Focal Loss | 损失降低 |
|-----------|---------|------------|--------|
| ŷ=0.9（易分类负样本） | 0.105 | 0.000263 | **99.75%** |
| ŷ=0.5（中等） | 0.693 | 0.043 | 93.8% |
| ŷ=0.1（难分类正样本） | 2.303 | 0.466 | 79.75% |

**结论**：Focal Loss 将 99.75% 的 easy sample 梯度贡献压缩至接近零，训练信号集中于难分类样本（真实信源位置），可期望在不均衡场景下将 F1 进一步提升 5~15%（需实际训练验证）。

---

### 11.5 CRB 理论下界与工程操作点

θ=0°、P=12（ULA 近似）条件下 T vs SNR 操作点建议：

| SNR (dB) | T=8 (°) | T=16 (°) | T=32 (°) | T=64 (°) |
|----------|---------|---------|---------|----------|
| -5 | 0.606 | 0.429 | 0.303 | 0.214 |
| 0 | 0.341 | 0.241 | 0.170 | 0.121 |
| **5** | **0.214** | **0.152** | **0.107** | **0.076** |
| 10 | 0.135 | 0.096 | 0.068 | 0.048 |
| 15 | 0.085 | 0.061 | 0.043 | 0.030 |

**规律性结论**：T 翻倍 ≡ SNR+3dB（等效关系），体现于对数轴上等间距平行线。

**推荐工程操作点**：SNR≥5dB、T=16 时 RMSE < 0.15°，满足大多数应用精度需求；低 SNR（<0dB）优先增加 T 而非传感器数，性价比更高。

---

### 11.6 综合结论与下一步建议

#### 已验证的关键结论

1. **TCA 有效性**：12 传感器 TCA 提供 69 自由度，是同等 ULA 的 6.3 倍
2. **MUSIC 精确性**：T=16、SNR=10dB 条件下误差 ~10⁻¹⁵°，完全收敛
3. **MDL 公式修正有效**：使用 AM-GM 噪声特征值比的 Wax-Kailath 复数公式，T=16 即可准确估计 K=3
4. **阈值优化收益显著**：0.5→0.75，F1 提升 57%，零计算开销

#### 待完成工作

| 项目 | 当前状态 | 建议 |
|------|---------|------|
| CNN 训练 | 仅示意曲线 | 用 TCA 虚拟协方差作为输入特征 |
| Focal Loss 实际效果 | 理论分析 | 替换 BCE 重新训练对比 |
| 低 SNR 鲁棒性 | CRB 分析 | SNR<0dB 时 T 至少需 64 |
| 真实阵列扰动 | ε=0.05 失配分析 | 扰动 5% 时失配误差 ~π²ε²/6 |

---

## 十、参考文献


1. M. Wax & T. Kailath, "Detection of signals by information theoretic criteria," IEEE Trans. ASSP, 1985.
2. T.-Y. Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017.
3. Z. Zheng, W.-Q. Wang et al., "DOA Estimation with CNN," IEEE MLSP 2020.
4. P. Pal & P. P. Vaidyanathan, "Nested Arrays," IEEE Trans. SP, 2010.
