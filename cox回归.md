在生物医学统计的考试中，**Cox 比例风险回归模型（Cox Proportional Hazards Model）** 是比 Kaplan-Meier 更进阶的分析方法。KM 曲线只能看一组因素（如药物 A vs B），而 **Cox 分析可以同时研究多个因素（年龄、性别、血压、用药量等）对生存时间的影响。**

在 Python 中，分析生存数据最常用的库是 `lifelines`。以下是考试标准的解题流程：

---

### 第一步：准备数据 (Data Setup)

Cox 模型要求数据包含：
1.  **生存时间 (Duration/Time)：** 连续变量。
2.  **事件状态 (Event/Status)：** 二分类变量（1 = 发生事件/死亡，0 = 删失/存活）。
3.  **协变量 (Covariates)：** 你想分析的影响因素（年龄、性别、治疗组等）。

```python
import pandas as pd
from lifelines import CoxPHFitter

# 假设数据 df 包含 columns: ['time', 'death_event', 'age', 'treatment', 'gender']
# 注意：Cox模型不能处理文本，必须把分类变量转为 0/1（Dummy Variables）
df_dummy = pd.get_dummies(df, columns=['treatment', 'gender'], drop_first=True)
```

---

### 第二步：检查比例风险 (PH) 假设 —— **考试加分/必考点**

Cox 模型有一个前提：**不同组别之间的风险比（Hazard Ratio）随时间是不变的。** 换句话说，生存曲线不能交叉。

在 Python 中，通过检查 **Schoenfeld 残差** 来验证：
```python
cph = CoxPHFitter()
cph.fit(df_dummy, duration_col='time', event_col='death_event')

# 检查假设
cph.check_assumptions(df_dummy, p_value_threshold=0.05)
```
*   **判断准则：** 如果 P 值 > 0.05，说明满足假设，可以继续。如果 < 0.05，则违反假设，考试时需注明。

---

### 第三步：拟合模型并输出摘要 (Model Fitting & Summary)

这是考试截屏的核心内容：
```python
# 拟合模型
cph.fit(df_dummy, duration_col='time', event_col='death_event')

# 输出统计表
cph.print_summary() 
```

---

### 第四步：解读统计表 (Crucial Interpretation)

运行 `print_summary()` 后，你需要盯着表中的这三列：

#### 1. exp(coef) —— 风险比 (Hazard Ratio, HR)
这是 Cox 分析的**灵魂**。
*   **HR > 1：** 危险因素。例如 `age` 的 HR = 1.05，表示年龄每增加一岁，死亡风险增加 5%。
*   **HR < 1：** 保护因素。例如 `treatment_Drug` 的 HR = 0.60，表示用药组的死亡风险比对照组降低了 40% ($1 - 0.60$)。
*   **HR = 1：** 无影响。

#### 2. p (P-value)
*   **$P < 0.05$：** 该因素对生存时间有**显著影响**。
*   **$P \ge 0.05$：** 该因素的影响在统计上不显著。

#### 3. Concordance (一致性指数)
*   类似 Logistic 回归的 AUC。通常在 0.6 - 0.7 表示模型还可以，> 0.8 表示预测非常准。

---

### 第五步：可视化 (Visualization)

考试中，如果能画出一张 **森林图 (Forest Plot)** 或 **调整后的生存曲线**，分数会更高。

```python
import matplotlib.pyplot as plt

# 1. 绘制森林图（显示每个变量的 HR 和置信区间）
cph.plot()
plt.title("Hazard Ratios with 95% Confidence Intervals")
plt.show()

# 2. 绘制对比图（比如在其他变量固定时，不同治疗方案的预测生存曲线）
cph.plot_partial_effects_on_outcome(covariates='treatment_Drug', values=[0, 1])
plt.show()
```

---

### 第六步：撰写考试结论 (Final Conclusion)

**标准答题模板：**
1.  **模型建立：** “构建了 Cox 比例风险回归模型，分析了年龄、性别、治疗方案对患者生存时间的影响。”
2.  **假设检验：** “经 Schoenfeld 残差检验，所有变量均满足比例风险假设（P > 0.05）。”
3.  **结果解读：** 
    *   “**治疗方案**是显著的保护因素（$HR = 0.45, P = 0.002$），说明新药能显著降低 55% 的死亡风险。”
    *   “**年龄**是显著的危险因素（$HR = 1.08, P = 0.01$），风险随年龄增长而升高。”
4.  **最终结论：** “在排除年龄和性别干扰后，该新药对延长患者寿命具有统计学意义上的显著效果。”

---

### 考试避坑指南：
*   **分类变量：** 别忘了 `pd.get_dummies`。直接把 "Male"/"Female" 扔进模型会报错。
*   **Status：** 确认你的 `event_col` 中 `1` 代表的是“死亡/事件发生”。如果 `1` 代表“存活”，HR 的解释就会全部反过来。
*   **截屏提示：** 上科大这种考试要求代码和结果在同一张图。确保你的 `cph.print_summary()` 结果完整显示在屏幕上再截图。