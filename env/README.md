对于智能体 $i$，其总奖励 $R_i$ 由以下几部分组成：

1. **局部速度对齐奖励 (40%)**：
```math
R_{align,i} = \frac{1}{|N_i|} \sum_{j \in N_i} \frac{\vec{v_i} \cdot \vec{v_j}}{|\vec{v_i}||\vec{v_j}|}
```
其中 $N_i$ 是智能体 $i$ 的邻居集合。这个值在 $[-1, 1]$ 范围内。

2. **局部聚集奖励 (30%)**：
```math
R_{cohesion,i} = \frac{1}{|N_i|} \sum_{j \in N_i} \exp\left(-\frac{(|\vec{r_{ij}}| - d_{opt})^2}{2\sigma^2}\right)
```
其中：
- $\vec{r_{ij}} = \vec{x_j} - \vec{x_i}$ 是智能体间的位置差
- $d_{opt} = 0.5R$ 是最优距离（R是通信半径）
- $ \sigma = 0.2R $ 是高斯函数的标准差
这个值在 [0, 1] 范围内。

3. **全局序参量 (20%)**：
```math
R_{global,i} = \frac{\vec{v_i} \cdot \vec{V}}{|\vec{v_i}||\vec{V}|}, \quad \vec{V} = \frac{1}{N}\sum_{j=1}^N \vec{v_j}
```
这个值在 $[-1, 1]$ 范围内。

4. **平滑控制惩罚 (10%)**：
```math
R_{smooth,i} = -0.1|\Delta \theta_i|
```
其中 $\Delta \theta_i$ 是转向角。这个值在 $[-0.1\pi, 0]$ 范围内（假设角度变化限制在 $±\pi$）。
