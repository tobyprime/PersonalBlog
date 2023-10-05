---
title: 最优化 - 二次规划
date: '2023-04-16 23:31'
tags:
  - Optimization
  - Math
categories:
  - 最优化
abbrlink: 42224
---
目标函数是变量的二次函数，约束条件是变量的线性不等式：
$$
\begin{aligned}
\min_{x} \quad &\frac{1}{2} x^T Q x + c^T x \\
s.t.\quad &Ax \leq b
\end{aligned}
$$
其中，$x\in\mathbb{R}^n,c\in\mathbb{R}^n,b\in\mathbb{R}^n,A\in\mathbb{R}^{m \times n}$，$Q\in\mathbb{R}^{n \times n}$是一个对称矩阵。
例如：$x=[x_{1},x_{2}]^T$,$Q=\left(\begin{matrix}a_{1} \quad a_{2}\\a_{2} \quad a_{3}\end{matrix}\right)$
$$
\begin{aligned}
\frac{1}{2}x^TQx
&=\frac{1}{2}\left(\begin{matrix}
x_{1} \quad x_{2}
\end{matrix}\right)
\left(\begin{matrix}
a_{1} \quad a_{2} \\
a_{2} \quad a_{3}
\end{matrix}\right)
\left(\begin{matrix}
x_{1} \\
x_{2}
\end{matrix}\right)
\\
&=\frac{1}{2}\left(\begin{matrix}
a_{1}x_{1}+a_{2}x_{2} \quad a_{2}x_{1}+a_{3}x_{2}
\end{matrix} \right) 
\left(
\begin{matrix}
x_{1} \\
x_{2}
\end{matrix}
\right)\\
&=\frac{1}{2}(a_{1}x_{1}^2+a_{2}x_{1}x_{2}+a_{2}x_{1}x_{2}+a_{3}x_{2}^2)\\
&=\frac{1}{2}a_{1}x_{1}^2+a_{2}x_{1}x_{2}+\frac{1}{2}a_{3}x_{2}^2
\end{aligned}
$$
其中的$\frac{1}{2}$是为了方便求导。

当$Q$为正定矩阵时，即$x^TQx>0$，$x^TQx$随着$|x|$的增加而增加，显然是严格凸的，而$c^Tx$是线性函数。此时该问题为严格凸二次规划问题，若可行域不为空，目标函数在此可行域有下界，则该问题有全局最小值。

$Q$为半正定矩阵时，为凸二次规划，有多组全局最优解。

$Q$为非正定矩阵时，是有多个平稳点和局部最优解的NP问题。