---
title: 最优化 - 拉格朗日乘子法
date: 2023-04-15 11:31
tags:
- Optimization
- Math
categories:
- 最优化
---
是一种寻找多元函数在**一组约束下**的极值的方法。通过引入拉格朗日乘子，将原问题的约束条件吸收进目标函数中形成新的函数，简化为**无约束优化问题**以方便求解：
1.  构造拉格朗日函数 $L(x,y,\dots,\lambda) = f(x,y,\dots) + \lambda g(x,y, \dots)$，其中 $f(x,y,\dots)$ 是原问题目标函数，$g(x,y, \dots)$$ 是约束条件。
2.  求解方程组 $\frac{ \partial L }{ \partial x }=0,\frac{ \partial L }{ \partial y }=0,\dots,\frac{ \partial L }{ \partial \lambda }=0$，得到所有可能的极值点 $(x,y,…,λ)$。
3.  将极值点代入目标函数 $f(x,y,\dots)$，比较大小，得到最大值和最小值。

假设要求 $f(x,y) = x^2 + y^2$ 在约束条件 $x + y = 1$ 下的最小值。：

1.  构造拉格朗日函数$L(x,y,\lambda) = f(x,y) + \lambda g(x,y)$，其中 $g(x,y) = x + y - 1$ 是约束条件转换而来。
2.  将偏导置零得到方程组 $\partial L/\partial x = 2x + \lambda = 0,\partial L/\partial y = 2y + \partial = 0,\partial/\partial \lambda L = x + y - 1 = 0$
3.  解得  $\lambda = -2,x = y = 1/2$，由于只有一组解，所以即为全局最优解。

对于不等式约束，需要通过 KKT 条件判断一个点是否是最优点。对于问题：
$$
\begin{aligned}
\min \quad &f(x)\\
s.t. \quad &g_{i}(x)\leq 0,i=1,\dots,m\\
&h_{j}(x)=0,j=1,\dots,n
\end{aligned}
$$
可以构造一个拉格朗日函数：
$$
L(x,\lambda,\mu)=f(x)+\sum_{i=1}^n\lambda_{i}g_{i}(x)+\sum_{j=1}^n\mu_{j}h_{j}(x)
$$
对于一组解 $x^\star,\lambda^\star,\mu^\star$，使用KKT 条件判断是否是一个最优解：
- 稳定性条件 ：$\nabla L(x^\star,\lambda^\star,\mu^\star)=0$
- 原始可行性：$g_{i}(x^\star)\leq 0;h_{j}(x^\star)= 0$
- 对偶可行性：$\lambda^\star_{i}\geq 0$
- 互补松弛可行性：$\lambda_{i}^\star g_{i}(x^\star)=0$

[知乎](https://zhuanlan.zhihu.com/p/556832103)
## 对偶可行性
![](https://pic2.zhimg.com/80/v2-052b0104b46e31fa4d7a05e9c6f3d2b5_720w.webp)
$x^\star$ 若是一个最优解，在 $x^\star$ 处，$g(x)$ 与 $f(x)$ 的梯度方向应该共线且相反，由稳定性条件可得
$\nabla L(x^\star,\lambda^\star,\mu^\star)=\nabla f(x^\star)+\sum_{i}\lambda_{i}\nabla g_{i}(x^{\star})=0\implies \lambda\nabla g(x)=-\nabla f(x)$即$\lambda g(x)$与$f(x)$梯度方向也相反，所以这里$\lambda\geq 0$否则不满足$g(x)$与$f(x)$。

由于条件成立时，$h_{j}(x^\star)=0$，且$g_{i}(x^\star)\leq 0$
$$f(x^\star)= g(\lambda^\star,\mu^\star)\leq f(x)+\sum_{i=1}^n\lambda_{i}^\star g_{i}(x^\star)+0$$
## 松弛互补可行性

$x^\star$ 有 $g_{i}(x^\star)< 0$时，即正好满足约束，此时**约束并不起作用**，所以$\lambda_{i}^\star=0$。
$x^*$ 有 $g_{i}(x^\star)=0$ 时，即可以转换为等号约束条件。
而对于 $g(x^\star)>0$的情况，不满足约束，需要舍弃。
在这两种满足约束的情况下$\lambda_{i}^\star g_{i}(x^\star)$恒等 于$0$，即为松弛互补可行性。

# 拉格朗日对偶问题
把 $L(x,\lambda,\mu)$ 看作是关于 $\lambda,\mu$ 的函数，求其最小值为拉格朗日对偶函数：
$$\theta(x)=\min_{x} L(x,\lambda,\mu)$$
也可以表达为$\theta(x)=\inf_{x \in \mathbb{D}}L(x,\lambda,\mu)$
则：
$$
\begin{aligned}
\max_{\lambda,\mu;\lambda\geq 0} \quad &\theta(\lambda,\mu)\\
s.t. \quad &\lambda\geq 0
\end{aligned}
$$
称为原问题的拉格朗日对偶问题，常通过对$x$求导并置零来获得对偶函数的表达形式（KKT的稳定性条件）。

若
$$
\max_{\lambda,\mu;\lambda\geq 0}\theta(\lambda,\mu)\leq \min_{x} \max_{\lambda,\mu;\lambda\geq 0}L(x,\lambda,\mu)
$$
则称为弱对偶，若
$$
\max_{\lambda,\mu;\lambda\geq 0}\theta(\lambda,\mu)= \min_{x} \max_{\lambda,\mu;\lambda\geq 0}L(x,\lambda,\mu)
$$
称为强对偶。

无论原问题是否为凸，对偶问题总是凸的，因为是关于$\lambda$和$\mu$的仿射函数。