---
title: SA-Siam
cover: structure.png
tags:
  - Object Tracking
  - Computer Vision
categories:
  - Paper Reading
abbrlink: 14313
date: 2022-11-08 08:16:33
---

# 简介
![](structure.png)
- 本文基于SiamFC，主要提升SiamFC的泛化能力
- SA-Siam 拥有两个分支
    - 外观（Appearance）分支：与SiamFC一致。
    - 语义（Semantic）分支：输出特征是high-level，健壮的，不易受外观影响，使模型能够剔除无关的背景，可以补充外观特征。
- 用于语义分支的通道注意力机制：对在特定目标中发挥更重要作用的通道给予更高的权重。

# 方法
## Semantic Branch
### 对于搜索
输入图像 $X$ 经过一个CNN（AlexNet）$f_s(\cdot)$，将最后两层的特征拼接以获得不同层次的信息。然后使用 $1 \times 1$ 卷积网络 $g(\cdot)$ 在相同层中融合特征。
$$ g(f_s(X)) $$
### 对于目标
不直接使用目标模板$z$，而使用以目标模板 $z$ 为中心，与搜索输入$X$一样大的图像 $z^s$ 作为CNN的输入（以获得更多上下文信息）得到 $f_s(z^s)f$。同时以 $f_s(z)$ 来表示$f_s(z^s)$ 经过裁剪（到以目标模板 $z$ 为输入的特征大小）的特征。 先将$f_s(z^s)$ 馈送到通道注意力模块得到通道权重 $\xi$，与裁剪后得到的 $f_s(z)$ 逐元素相乘，最后使用 $g(\cdot)$ 进行融合。
$$ g(\xi \cdot f_s(z)) $$
### Response Map
热力图（响应图）可以写为：
$$h_s(z^s,X) = corr(g(\xi \cdot f_s(z)),g(f_s(X)))$$
$corr(\cdot)$ 是相关运算。
## Appearance Branch
与 SiamFC 一样，表示为
$$h_a(z,X)=corr(f_a(z),f_z(X))$$
（这里不使用多级特征和添加通道关注是因为高级语义特征非常稀疏，而外观特征则相当密集，无法有效提升性能）
## 融合两个分支
仅在测试时进行融合，训练时是分别训练的。
$$h(z^s,X) = \lambda h_a(z,X)+(1-\lambda)h_s(z^s,X)$$
$\lambda$ 超参数平衡两个分支的重要性。

# 训练
**两个分支分别训练**
- Appearance 分支从头开始训练。
- Semanitc 分支使用冻结的预训练好的AlexNet（不进行微调，因为使用与A-Net相同的训练模式对S-Net进行微调会导致两个分支同质化），只训练融合模块。

{% btn 'https://arxiv.org/abs/1802.08817',Arxiv,fa fa-file-pdf,blue %}
{% btn 'https://github.com/microsoft/SA-Siam',Github,fa fa-code,red %}