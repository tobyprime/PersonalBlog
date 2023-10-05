---
title: 梯度下降
tags:
  - Machine Learnng
categories:
  - Machine Learnng
abbrlink: 54465
date: 2023-03-31 21:41:03
---
$$x^{t+1} := x^t - \eta \nabla f(x^t)$$
为了让 $f(x)$ 的值越来越小，需要不断迭代优化 $x$ 的值，$x$ 优化是方向是 $f$ 的梯度的反方向，快慢受到学习率 $\eta$  的影响。

# 梯度下降的本质
梯度下降的本质是，将目标函数 $f(x)$ 近似为其在某点 $x_{t}$ 处的一阶泰勒展开，记为 $\bar{f}(x;x_{t})$ ：
$$f(x) \approx \bar{f}(x;x_{t})=f(x_{t})+f^\prime(x^t)(x-x^t)$$
注意，只有在 $x_{t}\to x$ 时误差 $f(x)-\bar{f}(x;x_{t})$ 才趋近于 0，所以，我们要在 $x_{t}$ 附近找一点 $x_{t+1}=x_{t}+\Delta$ ，代入 $\bar{f}$ 可得：
$$\bar{f}(x_{t}+\Delta ;x_{t})=f(x_{t})-f^\prime(x_{t}) \cdot \Delta$$
我们希望，$f(x_{t+1})<f(x_{t})$ ，也就是 $\bar{f}(x_{t+1};x_{t})<f(x_{t})$ ，观察上式可以发现，只要令  $f^\prime(x_{t})\cdot \Delta <0$ 就能保证   $\bar{f}(x_{t+1};x_{t})<f(x_{t})$ 。

其中 $\Delta$ 是可以由我们自由定义的，不妨令 $f^\prime(x_{t})\cdot \Delta=-(f^\prime(x_{t}))^2$ ，此时 $\Delta=-f^\prime(x_{t})$，又因为误差，我们不希望 $\Delta$ 过大，可以再加入学习率 $\eta$ 来限制，此时
$$x_{t+1}=x_{t}+\Delta=x_{t}-\eta f^\prime(x_{t})$$此时再将 $f(x)$ 在 $x_{t+1}$ 处近似展开，继续求得 $x_{t+2}$ 的值，不断迭代更新。


