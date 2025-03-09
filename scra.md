

<figure><figcaption align="left">**Table 1** Different runs' result</figcaption></figure>

|                         | Run 0  | Run 1  | Run 2  | Run 3  | Run 4  | Run 5  | Run 6  | Run 7  | Run 8  | Run 9  |
| :---------------------- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- | :----- |
| **Test F1**             | 0.8339 | 0.8524 | 0.8266 | 0.8487 | 0.8524 | 0.8450 | 0.8450 | 0.8339 | 0.8598 | 0.8376 |
| **Training Time (s)**   | 6.3709 | 6.3226 | 5.9766 | 6.1177 | 6.3600 | 6.7334 | 6.4672 | 6.7090 | 6.2913 | 7.1234 |
| **Evaluation Time (s)** | 0.0480 | 0.0400 | 0.0480 | 0.0531 | 0.0460 | 0.0465 | 0.0475 | 0.0467 | 0.0410 | 0.0412 |

### 总体实验结果

|         指标          | 值             |
| :-------------------: | :------------- |
|      F1分数均值       | 0.8417         |
|     F1分数标准差      | 0.0110         |
|   平均模型训练时间    | 6.3401 seconds |
| 平均GIF单轮反训练时间 | 0.1676 seconds |









| 运行编号 | 训练时间 (秒) | 测试 F1 分数 | 测试准确率 | 测试召回率 | 测试精确率 |
| :------- | :------------ | :----------- | :--------- | :--------- | :--------- |
| 0        | 6.549         | 0.8339       | 0.8339     | 0.8339     | 0.8339     |
| 1        | 6.336         | 0.8303       | 0.8303     | 0.8303     | 0.8303     |
| 2        | 6.471         | 0.8450       | 0.8450     | 0.8450     | 0.8450     |

|      |
| ---- |

​		

| 指标                    | 平均值 | 标准差 |
| :---------------------- | :----- | :----- |
| F1 分数                 | 0.8364 | 0.0063 |
| 准确率                  | 0.8364 | 0.0063 |
| 召回率                  | 0.8364 | 0.0063 |
| 精确率                  | 0.8364 | 0.0063 |
| 模型训练时间 (秒)       | 6.452  | -      |
| GIF F1 分数             | 0.8290 | 0.0046 |
| GIF 准确率              | 0.8413 | 0.0030 |
| GIF 召回率              | 0.8413 | 0.0030 |
| GIF 精确率              | 0.8413 | 0.0030 |
| GIF unlearning时间 (秒) | 0.2236 | -      |





在GNN中，两个互为邻居的节点a和b之间的相互影响可以通过影响函数（influence function）建模。假设a对b的影响为\( f \)，b对a的影响为\( g \)，其关系可基于以下步骤推导：

---

### **1. 影响函数的数学定义**

在节点遗忘（Node Unlearning）任务中，影响函数通常与参数更新方向相关。根据提供的公式（20），参数更新\( $\Delta \Theta $\)可分解为两部分：

- **第一部分**：包含节点\( $\mathcal{V}(rm) $\)及其邻居\($ N_k(\mathcal{V}(rm))$ \)的梯度贡献。
- **第二部分**：移除邻居节点后的梯度修正。

具体地，节点间的相互影响可通过Hessian矩阵\($ H_{\theta_0}^{-1}$ \)和梯度项\( $\nabla_{\theta_0} l$ \)表达。假设：

- \( f \)对应a对b的影响：  
  $f = H_{\theta_0}^{-1} \cdot \nabla_{\theta_0} l\left( f_\Theta(a, y_a) \right),$
- \( g \)对应b对a的影响：  
  $g = H_{\theta_0}^{-1} \cdot \nabla_{\theta_0} l\left( f_\Theta(b, y_b) \right).$

---

### **2. 从\( f \)推导\( g \)（或反之）的对称性**

在GNN中，节点间的消息传递具有对称性（无向图）或方向性（有向图）。若图为无向图，则影响函数满足以下关系：

#### **（1）基于梯度的对称性**

由于a和b互为邻居，两者的梯度可通过链式法则关联。假设损失函数\( l \)对节点表征的导数满足：
$\[
\nabla_{\theta_0} l\left( f_\Theta(a) \right) = \frac{\partial l}{\partial f_\Theta(a)} \cdot \frac{\partial f_\Theta(a)}{\partial \Theta},
\]
\[
\nabla_{\theta_0} l\left( f_\Theta(b) \right) = \frac{\partial l}{\partial f_\Theta(b)} \cdot \frac{\partial f_\Theta(b)}{\partial \Theta}.
\]$

当节点a和b的特征通过聚合相互影响时，存在：
$\[
\frac{\partial f_\Theta(a)}{\partial \Theta} = \text{Aggregate}\left( \frac{\partial f_\Theta(b)}{\partial \Theta} \right),
\]
\[
\frac{\partial f_\Theta(b)}{\partial \Theta} = \text{Aggregate}\left( \frac{\partial f_\Theta(a)}{\partial \Theta} \right).
\]$

因此，\( f \)和\( g \)可通过聚合操作相互转换。例如，若聚合函数为均值，则：
$\[
g = \frac{1}{|\mathcal{N}(a)|} \cdot f, \quad f = \frac{1}{|\mathcal{N}(b)|} \cdot g.
\]$

#### **（2）基于Hessian逆的共享性**

由于Hessian矩阵\( H_{\theta_0}^{-1} \)是全局的（与所有节点相关），而非节点特异性，因此\( f \)和\( g \)的转换仅依赖于梯度项的对称性：
\[
g = H_{\theta_0}^{-1} \cdot \left( \nabla_{\theta_0} l(b) \right) = H_{\theta_0}^{-1} \cdot \left( \nabla_{\theta_0} l(a) \cdot J_{a \to b} \right),
\]
其中\( J_{a \to b} \)是a到b的雅可比矩阵，反映节点表征的传播路径。

---

### **3. 具体推导过程**

假设在单层GNN中，节点表征更新公式为：
\[
h_a^{(l)} = \sigma\left( W \cdot \text{Mean}\left( \{ h_b^{(l-1)} \mid b \in \mathcal{N}(a) \} \right) \right),
\]
则损失对参数\( W \)的梯度为：
\[
\nabla_W l = \sum_{a} \frac{\partial l}{\partial h_a^{(l)}} \cdot \frac{\partial h_a^{(l)}}{\partial W}.
\]

对于互为邻居的a和b，其梯度贡献分别为：
\[
\nabla_W l(a) = \frac{\partial l}{\partial h_a^{(l)}} \cdot \sigma'(\cdot) \cdot \text{Mean}\left( \{ h_b^{(l-1)} \} \right),
\]
\[
\nabla_W l(b) = \frac{\partial l}{\partial h_b^{(l)}} \cdot \sigma'(\cdot) \cdot \text{Mean}\left( \{ h_a^{(l-1)} \} \right).
\]

因此，\( f \)和\( g \)的关系可表示为：
\[
g = \frac{\partial l}{\partial h_b^{(l)}} \cdot \left( \nabla_W l(a) \right)^T \cdot H_{\theta_0}^{-1},
\]
\[
f = \frac{\partial l}{\partial h_a^{(l)}} \cdot \left( \nabla_W l(b) \right)^T \cdot H_{\theta_0}^{-1}.
\]

---

### **4. 最终结论**

- **从\( f \)到\( g \)**：  
  通过交换梯度来源节点，并利用Hessian逆的共享性，可得：
  \[
  g = H_{\theta_0}^{-1} \cdot \nabla_{\theta_0} l(b) = f \cdot \left( \frac{\nabla_{\theta_0} l(b)}{\nabla_{\theta_0} l(a)} \right).
  \]
- **从\( g \)到\( f \)**：  
  反向操作，交换节点角色：
  \[
  f = H_{\theta_0}^{-1} \cdot \nabla_{\theta_0} l(a) = g \cdot \left( \frac{\nabla_{\theta_0} l(a)}{\nabla_{\theta_0} l(b)} \right).
  \]

---

### **5. 实际应用中的简化**

若假设梯度的比例关系由节点度数决定（如GCN的归一化），则更简洁的关系为：
$
g = f \cdot \frac{|\mathcal{N}(a)|}{|\mathcal{N}(b)|}, \quad f = g \cdot \frac{|\mathcal{N}(b)|}{|\mathcal{N}(a)|}.
$

这一关系体现了节点度数对相互影响的调节作用。