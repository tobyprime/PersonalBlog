---
title: 支持向量机
date: 2023-04-06 11:54:33
cover: cover.png
tags:
- Machine Learnng
categories:
- Machine Learnng
---
是一种二分类模型，目的是找到一个超平面，使得它能够正确划分训练数据集，并且使得训练数据集中离超平面最近的点（即支持向量）到超平面的距离最大。

# 硬间隔SVM
![](hard-svm.png)

定义有三个超平面：
1. 超平面： $w^Tx+b=0$，这个超平面用于在预测时，判断在两个超平面之间的样本点。
2. 正超平面：$w^Tx+b=1$，优化时，保证正类都在其之上
3. 负超平面：$w^Tx+b=-1$，优化时，保证负类都在其之下

样本中任意点到超平面$w^Tx+b=0$的距离可以写为：
$$
r_{i}=\frac{|w^Tx+b|}{\Vert  w \Vert }
$$

假设正超平面到超平面的距离为 $r^+$
$$
\begin{cases}
w^Tx+b=1 \\
\frac{|w^Tx+b|}{\Vert w\Vert }=r^+
\end{cases}
$$
解得
$$
r^+=\frac{1}{\Vert w\Vert }
$$
在我们的定义中正负两超平面对称，$r^+=r^-$，即正负超平面之间的间隔 $r=2r^+$：
$$r=\frac{2}{\Vert w\Vert }\tag{1}$$
此时若能正确分类则任意的样本点$(x^{(i)},y^{(i)})$满足：
$$
\begin{cases}
w^Tx^{(i)}+b\geq 1&,y^{i}=1 \\
w^Tx^{(i)}+b\leq -1&,y^{i}=-1
\end{cases}
$$
由于，标签$y^{(i)} \in(+1,-1)$，如果模型预测正确，则预测值的符号与标签符号相等，积大于$1$
所以可以将上式简写为：
$$y^{(i)}(w^Tx^{(i)}+b)\geq 1 \tag{2}$$

为了模型的泛化性能，我们希望，间隔 $r$ 越大越好：
$$ 
\begin{aligned}
\max_{w,b} r &= \min_{w,b} \frac{1}{r}\\
&=\min_{w,b} \frac{1}{2}\Vert w\Vert \\
&=\min_{w,b} \frac{1}{2}\Vert w\Vert ^2\\
&=\min_{w,b} \frac{1}{2}w^Tw
\end{aligned} 
$$
此时优化目标可以描述为一个**凸二次规划**问题
$$
\begin{aligned}
&\min_{w,b} \Vert w\Vert ^2\\
&s.t. \ \ \forall i \ \ \ y^{(i)}(w^Tx^{(i)}+b)\geq 1 \ \ \text{and} \ \ \zeta_{i}\geq 0
\end{aligned}\tag{3}
$$
注意，硬间隔无法解决线性不可分的问题，因此，需要迎入**软间隔**。

# 软间隔
很多情况下，数据是线性不可分的，即，永远无法满足式$(2)$，此时我们需要引入铰链损失：
$$\zeta_{i}=\max(0,1-y^{(i)}(w^Tx^{(i)}+b)) \tag{4}$$
即，若样本点 $x^{(i)}$ 被正确分类时，损失为 $0$，被错误分类时，损失的值与到超平面（并非正超平面或负超平面）的距离成正比。

对于目标$(1)$改为：
$$\lambda\sum_{i}\zeta_{i}+\frac{1}{2}\Vert w\Vert ^2$$

对于条件$(2)$改为：
$$y^{(i)}(w^Tx^{(i)}+b)\geq 1 -\zeta_{i} \tag{5}$$
即，对于被错误分类的样本点，不进行约束。
原问题改为：
$$
\begin{aligned}
&\min_{w,b} \frac{1}{2}\Vert w\Vert ^2+\lambda\frac{1}{n}\sum_{i}\zeta_{i}\\
&s.t. \ \ \forall i\ \ -y^{(i)}(w^Tx^{(i)}+b)-1+\zeta_{i}\leq0 \ \ \text{and} \ \ -\zeta_{i}\leq 0 \\
\end{aligned}\tag{6}
$$
$\lambda$ 表示了间隔大小的重要程度，仍然是凸优化问题强对偶性成立，其拉格朗日函数为：
$$
L(w,b,\alpha,\mu)=\frac{1}{2}\Vert w\Vert ^2+\lambda\sum_{i}\zeta_{i}-\sum_{i}\alpha_{i}\left[y^{(i)}(w^Tx^{(i)}+b)-1+\zeta_{i}\right]-\sum_{i}\mu_{i}\zeta_{i}
$$
分别对$w,b$求偏导并置$0$，由于这里对 $\zeta$ 求导比较复杂，可以直接$L$对 $\zeta$ 求导，因为
$$
f(x,y)=g(x,y,h(x,y))
$$
$$
\frac{\partial f}{\partial x}=\frac{\partial f}{\partial h}\frac{\partial h}{\partial x}
$$
若导数存在且 $\frac{\partial f}{\partial h}|_{h=h^\star}=0$，必然 $\frac{\partial f}{\partial x}|_{h=h^\star}=0$，意义相同。

所以拉格朗日函数对$w,b,\zeta$分别求偏导得到：
$$
\begin{aligned}
\frac{ \partial L }{ \partial w } &=w-\sum_{i}\alpha_{i} y^{(i)}x^{(i)}=0 \implies w=\sum_{i}\alpha_{i}y^{(i)}x^{(i)} \\
\frac{ \partial L }{ \partial b } &= \sum_{i}\alpha_{i} y^{(i)}=0\\
\frac{ \partial L }{ \partial \zeta } &=\lambda- \alpha-\mu=0 \implies \lambda-\mu=\alpha
\end{aligned}\tag{7}
$$
其中$\mathbb{1}\in \mathbb{R}^n,\forall i \ \ \mathbb{1}_{i}=1$，$\alpha=\{\alpha_{i}\}^n_{i=1}$，$\mu=\{\mu_{i}\}^n_{i=1}$

代回拉格朗日函数，得到拉格朗日对偶函数：
$$
\begin{aligned}
g(\alpha,\mu)&=\frac{1}{2} w^Tw-w^T\sum_{i}\alpha_{i}y^{(i)}x^{(i)}+\sum_{i}\alpha_{i}y^{(i)}b+\sum_{i}\alpha_{i}+\lambda\sum_{i}\zeta_{i}-\sum_{i}\alpha_{i}\zeta_{i}-\sum_{i}\mu_{i}\zeta_{i}\\

g(\alpha,\mu)&=w^T(w-\alpha_{i}y^{(i)}x^{(i)})-\frac{1}{2}w^Tw+b\sum_{i}\alpha_{i}y^{(i)}+\sum_{i}\alpha_{i}+\sum \left( \lambda-\alpha-\mu \right)\zeta\\

g(\alpha,\mu)&=-\frac{1}{2}w^Tw+\sum_{i}\alpha_{i}+\sum \left( \lambda-\alpha-\mu \right)\zeta\\

g(\alpha,\mu)&=-\frac{1}{2}\sum_{i}\sum_{j}\alpha_{i}y^{(i)}({x^{(i)}}^Tx^{(j)})\alpha_{j}y^{(j)}+\sum_{i}\alpha_{i}
\end{aligned}


$$


由于松弛互补可行性，需要满足：$$a_{i}y^{(i)}=0$$
由于式$(7.2)$与$(7.3)$：
$$
\begin{aligned}
\lambda-\alpha=\mu\geq 0\\
\alpha\leq \lambda
\end{aligned}
$$
并与对偶可行性融合：
$$0\leq \alpha\leq \lambda$$

则简化后的对偶问题表述为：
$$
\begin{aligned}
&\max_{w,b} -\frac{1}{2}\sum_{i}\sum_{j}\alpha_{i}y^{(i)}({x^{(i)}}^Tx^{(j)})\alpha_{j}y^{(j)}+\sum_{i}\alpha_{i}\\
&s.t. \ \ \forall i\ \ \alpha_{i}y^{(i)}=0 \ \ \text{and} \ \ 0\leq \alpha_{i}\leq \lambda
\end{aligned}
$$
转为最小化，并转为二次规划的一般形式：
$$
\max_{w,b} g(\alpha,\mu)=\min_{w,b}\frac{1}{2}\sum_{i}\sum_{j}\alpha_{i}y^{(i)}({x^{(i)}}^Tx^{(j)})\alpha_{j}y^{(j)}-\sum_{i}\alpha_{i}
$$
令$Q=\{Q_{i,j}\}_{i=1,j=1}^n$，原问题可描述为：
$$
\begin{aligned}
&\min_{w,b}\frac{1}{2}\alpha^TQ\alpha-\mathbb{1}^T\alpha \\
&s.t. \ \ \alpha^Ty=0 \ \ \text{and} \ \ 0\leq \alpha\leq \lambda
\end{aligned}\tag{8}
$$
以方便的使用二次规划算法解出，对于最优解$\alpha^\star$，原问题最优解：
$$
w^\star=\sum_{i}\alpha^\star_{i}y^{(i)}x^{(i)}
$$
$$y^{(i)}(w^Tx^{(i)}+b)=1\Longleftrightarrow b=\frac{1}{y^{(i)}}- w^Tx^{(i)}$$
由于$y^{(i)}\in{+1,-1}$，$\frac{1}{y^{(i)}}=y^{(i)}$
$$b^\star=\frac{1}{y^{i}}-{w^\star}^Tx^{(i)}$$
# 核方法
与线性回归的激活函数类似，通过一个非线性函数$\phi(\cdot)$来映射数据使得更容易线性可分。对于核函数的选择，是$SVM$最重要的问题，直接决定了是否能有效分类。
对称函数$\mathcal{K}(x^{(i)},x^{(j)})=\phi(x^{(i)})^T\phi(x^{(j)})$
对于数据集$D=\{x^{(1)},\dots,x^{(n)}\}$，$\mathcal{K}$的核矩阵：
$$K=\left[\begin{matrix}
\mathcal{K}(x_{1},x_{1}) \ \dots \ \mathcal{K}(x_{1},x_{n}) \\
\dots \\
\mathcal{K}(x_{n},x_{1}) \ \dots \mathcal{K}(x_{n},x_{n}
\end{matrix}\right]$$
是半正定的，则这个函数就能作为核函数使用，此时目标可以写为：
$$\min_{w,b}\frac{1}{2}\sum_{i}\sum_{j}\alpha_{i}y^{(i)}\mathcal{K}(x_{i},x_{j})\alpha_{j}y^{(j)}-\sum_{i}\alpha_{i}
$$

常见核函数：
- 线性核：${x^{(i)}}^Tx^{(j)}$
- 多项式核：$({x^{(i)}}^Tx^{(j)})$
- 高斯核：$\exp \left(-\frac{\Vert x^{(i)}-x^{(j)}\Vert^2 }{2\sigma^2}\right)$
- 拉普拉斯核：$\exp\left( -\frac{\Vert x^{(i)}-x^{(j)}\Vert }{2\sigma} \right)$
- Sigmoid 核：$\tanh(\beta {x^{(i)}}^Tx^{(j)}+\theta)$
