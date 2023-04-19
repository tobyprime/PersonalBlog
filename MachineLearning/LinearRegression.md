---
title: 线性回归
date: 2023-03-31 21:41:03
cover: cover.gif
tags:
- Machine Learnng
categories:
- Machine Learnng
---
# 普通线性回归 Linear Regression
## 一般形式

$$ f(x)=w_1x_1+...+w_kx_k+b $$
需要优化的参数为权重$w_n$与偏置$b$，通常使用最小二乘法估计模型参数。

一般写作向量形式：
$$f(x)=wx+b$$
其中 $w=(w_1,...,w_k)^T$，$x=(x_1,...,x_k)$


## 优化（最小二乘法）Least Square Method
目标是求出一组参数$w,b$，使得对于所有输入的预测值与输出值的 MSE 最小。
定义MSE为：
$$E = \sum_i^n(f(x_i)-y_i)$$
优化目标是
$$w^*,b^*=\argmin_{w,b} E$$

### 一元线性回归的最小二乘法推导
如下图所示，蓝色为预测值 $f(x)$，粉色为真实值 $y$，灰色阴影部分为 $E$，$E$ 在 $f(x)$ 与 $y$ 相交时为 $0$，而交点左右逐渐增大，可见 $E$ **为一个凸函数**。
![](E.png)

因为 $E$ 是凹函数，其导数为 0 时 $E$ 刚好为最小值，所以为了求一组 $w$ 与 $b$ 使得 $E$ 最小可以通过对 $E$ 分别求 $w$ 和 $b$ 的偏导并置 $0$ 得到闭式解：
$$
\begin{aligned}
\frac{\partial E}{\partial w} &= \frac{\partial}{\partial w}[\sum_i^n(wx_i+b-y_i)^2] \\
& = \sum_i^n\frac{\partial}{\partial w}[(wx_i+b-y_i)^2]\\
& = \sum_i^n[2\cdot(wx_i+b-y_i)(x_i)] \\
&=2\cdot(w\sum_i^nx^2_i-\sum_i^nx_iy_i+b\sum_i^nx_i)\\
&=2\cdot(w\sum_i^nx^2_i-\sum_i^n(y_i-b)x_i)
\end{aligned}
\tag{1}
$$
$$
\begin{aligned}
\frac{\partial E}{\partial b} & = \sum_i^n\frac{\partial}{\partial w}[(wx_i+b-y_i)^2]\\
& = \sum_i^n[2\cdot(wx_i+b-y_i)] \\
& = 2\cdot(nb - \sum_i^n(y_i-wx_i))
\end{aligned}
\tag{2}
$$
将式 $(1)$ 置 $0$：
$$
\begin{aligned}
0 &= 2\cdot(w\sum_i^nx^2_i-\sum_i^n(y_i-b)x_i)\\
0 &= w\sum_i^nx^2_i-\sum_i^n(y_i-b)x_i\\
w\sum_i^nx^2_i &= \sum_i^n(y_i-b)x_i
\end{aligned}
\tag{3}
$$
将式 $(2)$ 置 $0$：
$$
\begin{aligned}
0 & = 2\cdot(nb - \sum_i^n(y_i-wx_i))\\
b &= \frac{1}{n}\sum_i^n(y_i)-w(\frac{1}{n}\sum_i^nx_i)\\
b &= \bar{y}-w\bar{x}
\end{aligned}
\tag{4}
$$

将 $(4)$ 代入 $(3)$：
$$
\begin{aligned}
w \sum_i^n x^2_i &= \sum_i^n (y_i-(\bar{y}-w \bar{x}))x_i \\
w \sum_i^n x^2_i &= \sum_i^n(y_i-\bar{y}+w \bar{x})x_i \\
w\sum_i^n x^2_i &= \sum_i^n(x_iy_i-\bar{y}x_i+w\bar{x}x_i) \\
w\sum_i^n x^2_i &= \sum_i^n x_i y_i-\bar{y}\sum_i^n x_i+w\bar{x} \sum_i^n x_i \\
w(\sum_i^n x^2_i - \bar{x} \sum_i^n x_i) &= \sum_i^n x_i y_i-\bar{y}\sum_i^n x_i \\
w &= \frac{\sum_i^n x_i y_i-\bar{y}\sum_i^n x_i}{\sum_i^n x^2_i - \bar{x} \sum_i^n x_i} \\
w &= \frac{\sum_i^n x_i y_i-\frac{1}{n}\sum_i^n y_i \sum_i^n x_i}{\sum_i^n x^2_i - \frac{1}{n}\sum_i^n x_i \sum_i^n x_i} \\
w &= \frac{\sum_i^n x_i y_i-\sum_i^n y_i \bar{x}}{\sum_i^n x^2_i - \frac{1}{n}\sum_i^n x_i^2} \\
w &= \frac{\sum_i^n y_i (x_i-\bar{x})}{\sum_i^n x^2_i - \frac{1}{n}\sum_i^n x_i^2}
\end{aligned} 
\tag{5}
$$

将 $(5)$ 向量化以能够使用矩阵运算加速库
$$
\begin{aligned}

w &= \frac{\sum_i^n x_i y_i-\bar{y}\sum_i^n x_i}{\sum_i^n x^2_i - \bar{x} \sum_i^n x_i} \\
w &= \frac{\sum_i^n x_i y_i-\bar{y} (n \cdot \frac{1}{n} \sum_i^n{x_i})}{\sum_i^n x^2_i - \bar{x} (n\cdot \frac{1}{n}\sum_i^n x_i)} \\
w &= \frac{\sum_i^n x_i y_i-\bar{y} (n \bar{x})}{\sum_i^n x^2_i - \bar{x} (n \bar{x})} \\
w &= \frac{\sum_i^n x_i y_i-n\bar{x}\bar{y}}{\sum_i^n x^2_i - n \bar{x}^2} \\
w &= \frac{\sum_i^n x_i y_i-\sum_i^n\bar{x}\bar{y}}{\sum_i^n x^2_i - \sum_i^n \bar{x}^2} \\
w &= \frac{\sum_i^n (x_i y_i-(\bar{x}\bar{y}-x_i\bar{y}+\bar{x}y_i))}{\sum_i^n (x^2_i - (\bar{x}^2-x_i\bar{x}+\bar{x}x_i))} \\
w &= \frac{\sum_i^n (x_i y_i-(x_i\bar{y}+\bar{x}y_i-\bar{x}\bar{y}))}{\sum_i^n (x^2_i - (x_i\bar{x}+\bar{x}x_i-\bar{x}^2))} \\
w &= \frac{\sum_i^n (x_iy_i-x_i\bar{y}-\bar{x}y_i+\bar{x}\bar{y})}{\sum_i^n (x^2_i-x_i\bar{x}-\bar{x}x_i+ \bar{x}^2)} \\
w &= \frac{\sum_i^n (x_i(y_i-\bar{y})-\hat{x}(y_i-\bar{y}))}{\sum_i^n (x_i(x_i-\bar{x})-\bar{x}(x_i-\bar{x}))} \\
w &= \frac{\sum_i^n (x_i-\bar{x})(y_i-\bar{y})}{\sum_i^n (x_i-\bar{x})^2} \\
\end{aligned} 
$$
令 $x_d=(x_1-\bar{x};...;x_n-\bar{x})$，为去均值后的 $x$，$y_d=(y_1-\bar{y};...;y_n-\bar{y})$ 为去均值后的 $y$，代入上式：
$$
\begin{aligned}
w=\frac{x_d^Ty_d}{d_d^T x_d}
\end{aligned} 
$$


### 多元线性回归的推导
便于讨论，令
$$
X=
\left(\begin{matrix}
x_{11} & ... & x_{1n} & 1\\
... & ... & ... & ... \\
x_{n1} & ... & x_{nd} & 1
\end{matrix}\right) = 
\left(\begin{matrix}
x_1^T & 1\\
... & ... \\
x_n^T & 1
\end{matrix}\right) =
\left(\begin{matrix}
\hat{x}_1^T\\
... \\
\hat{x}_n^T
\end{matrix}\right) 
$$
$$
\hat{w}=(w;b)
$$
$$
\begin{aligned}
E&=\sum^n_i(\hat{x}_i^T \hat{w})^2\\
&=\left[\begin{matrix}y_1-\hat{x}_1^T \hat{w} &...& y_1-\hat{x}_1^T \hat{w}\end{matrix}
\right]
\left[\begin{matrix}y_1-\hat{x}_1^T \hat{w} \\ 
... \\
 y_1-\hat{x}_1^T \hat{w}
\end{matrix}\right]\\
&=(y-X\hat{w})^T(y-X\hat{w})
\end{aligned}
$$
将 $E$ 展开:
$$E=y^Ty-y^TX\hat{w}-\hat{w}^TX^Ty+\hat{w}^TX^TX\hat{w}$$
对 $\hat{w}$ 求导得：
$$
\frac{d E}{d\hat{w}}=\frac{d}{d\hat{w}}y^Ty-\frac{d}{d\hat{w}}y^TX\hat{w}-\frac{d}{d\hat{w}}\hat{w}^TX^Ty+\frac{d}{d\hat{w}}\hat{w}^TX^TX\hat{w}
$$
由[矩阵求导法](https://en.wikipedia.org/wiki/Matrix_calculus)则可得：
$$
\begin{aligned}
\frac{d E}{d\hat{w}}&=0-X^Ty-X^Ty+(X^Ty+X^TX)\hat{w}\\
&=2X^T(X\hat{w}-y)
\end{aligned} 
$$
凸函数证明过程略，将式 $(7)$ 置$0$
$$
\begin{aligned}
0&=2X^T(X\hat{w}-y) \\
X^TX\hat{w}&=X^Ty\\
\hat{w}&=(X^TX)^{-1}X^Ty
\end{aligned} 
$$
若$X^TX$为非正定矩阵时，可以引入正则项或使用伪逆矩阵，这都会导致出现多个解，使得没法直接求出全局最优解。

# 广义线性模型
让模型逼近 $y$ 的衍生物
$$ 
\begin{aligned}
g(y) &= w^Tx+b\\
y&=g^{-1}(w^Tx+b)
\end{aligned} 
$$
$g(\cdot)$中不应该有需要优化的参数。
## 对数几率回归 Logit Regression
通常用于二分类预测问题，预测 $y$ 为样本x作为正例的可能性，则$1-y$ 为反例的可能性，两者比值 $\frac{y}{1-y}$ 反应了 $x$ 作为正例的相对可能性，再对其取对数，得到对数几率 $\ln \frac{y}{1-y}$，对数几率回归：
$$
\begin{aligned}
\ln\frac{y}{1-y}&=w^Tx+b\\
y&=\frac{1}{1+e^{-(w^Tx+b)}}
\end{aligned}
$$
对于二分类的误差，通常不使用MSE而是BCELoss（Binary Cross Entropy）来衡量：
$$
\ell(y,\hat{y})=y\cdot\ln\hat{y}+(1-y)\ln(1-\hat{y})
$$
整体误差：
$$
\begin{aligned}
E &= -\sum^n_i(y\cdot\ln(\frac{1}{1+e^{-(\hat{w}^T\hat{x}_i)}})+(1-y)\ln(\frac{e^{-\hat{w}^T\hat{x}_i}}{1+e^{-\hat{w}^T\hat{x}_i}}))\\
&=-\sum^n_i(y\cdot\ln(\frac{1}{1+e^{-\hat{w}^T\hat{x}_i}})+\ln(\frac{e^{-\hat{w}^T\hat{x}_i}}{1+e^{-\hat{w}^T\hat{x}_i}})-y\cdot\frac{e^{-\hat{w}^T\hat{x}_i}}{1+e^{-\hat{w}^T\hat{x}_i}})\\
&=-\sum^n_i(-y\cdot\ln(1+e^{-(\hat{w}^T\hat{x}_i)})-\ln(1+e^{\hat{w}^T\hat{x}_i})+y\cdot\ln(1+e^{\hat{w}^T\hat{x}_i}))\\
&=-\sum^n_i(y\cdot[\ln(1+e^{\hat{w}^T\hat{x}_i})-\ln(1+e^{-\hat{w}^T\hat{x}_i})]-\ln(1+e^{\hat{w}^T\hat{x}_i}))\\
&=-\sum^n_i(y\cdot\ln(\frac{1+e^{\hat{w}^T\hat{x}_i}}{1+e^{-\hat{w}^T\hat{x}_i}})-\ln(1+e^{\hat{w}^T\hat{x}_i}))\\
&=-\sum^n_i(y\cdot\ln(\frac{1+e^{\hat{w}^T\hat{x}_i}}{1+\frac{1}{e^{\hat{w}^T\hat{x}_i}}})-\ln(1+e^{\hat{w}^T\hat{x}_i}))\\
&=-\sum^n_i(y\cdot\ln(\frac{(1+e^{\hat{w}^T\hat{x}_i})\cdot e^{\hat{w}^T\hat{x}_i}}{e^{\hat{w}^T\hat{x}_i}+\frac{1}{e^{\hat{w}^T\hat{x}_i}}\cdot e^{\hat{w}^T\hat{x}_i}})-\ln(1+e^{\hat{w}^T\hat{x}_i}))\\
&=-\sum^n_i(y\cdot\ln(\frac{(1+e^{\hat{w}^T\hat{x}_i})\cdot e^{\hat{w}^T\hat{x}_i}}{1+e^{\hat{w}^T\hat{x}_i}})-\ln(1+e^{\hat{w}^T\hat{x}_i}))\\
&=\sum^n_i(\ln(1+e^{\hat{w}^T\hat{x}_i})-y\cdot \hat{w}^T\hat{x}_i)\\
\end{aligned}
$$
目标为：
$$\hat{w}^*=\argmin_{\hat{w}}E$$
$E$为凸函数的证明略。最优化模型可以通过梯度下降、牛顿法等求得最优解。

### 对数几率回归的梯度下降推导
$$\hat{y}=\frac{1}{1+e^{-(wx+b)}}$$
$$=\sum^n_i(\ln(1+e^{\hat{w}^T\hat{x}_i})-y\cdot \hat{w}^T\hat{x}_{i)\\}$$
$E$ 对 $w_k$ 求偏导：
$$
\begin{aligned}
\frac{\partial E}{\partial w}&=\sum^n_i(\frac{\partial}{\partial w}\ln(1+e^{wx_{ik}+b})-\frac{\partial}{\partial w}y_i\cdot (wx_{ik}+b))\\
&=\sum^n_i(\frac{1}{1+e^{wx_{ik}+b}}\cdot e^{wx_{ik}+b}\cdot x_{ik}-y_i\cdot x_{ik}) \\
&=\sum^n_i(\frac{1}{1+\frac{1}{e^{-(wx_{ik}+b)}}}\cdot \frac{1}{e^{-(wx_{ik}+b)}}\cdot x_{ik}-y_i\cdot x_{ik}) \\
&=\sum^n_i(\frac{1}{e^{-(wx_{ik}+b)}+1}\cdot x_{ik}-y_i\cdot x_{ik}) \\
&=\sum^n_i(\hat{y_i}-y_i)\cdot x_{ik}

\end{aligned}
$$
$E$ 对 $b$ 求偏导：
$$
\begin{aligned}
\frac{\partial E}{\partial b}&=\sum^n_i(\frac{1}{e^{-(wx_{ik}+b)}+1}-y_i) \\
\frac{\partial E}{\partial b}&=\sum^n_i(\hat{y}-y_i) \\

\end{aligned}
$$

梯度下降迭代：
$$
w_k:= w_k -  \eta\sum^n_i(\hat{y_i}-y_i)\cdot x_{ik},k=1,....,d
$$
$$
b:= b -  \eta\sum^n_i(\hat{y}-y_i)
$$