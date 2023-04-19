---
title: SwinTrack
date: 2022-11-11 18:20:15
cover: structure.png
tags:
- Object Tracking
- Computer Vision
categories:
- Paper Reading
---
# 概述
- 基于 Siamese 框架的简单而高效的全注意力跟踪模型。
- 提出了motion token，编码了目标的历史轨迹，通过提供时间上下文来改进跟踪性能，且轻量不影响速度。
- SwainTrack 由三个部分组成：
    1. 用于特征提取的 Swin-Transformer主干网络。
    2. 用于视觉-运动特征融合的编码器-解码器网络。
    3. 用于分类与边界框回归的头部网络。

![](structure.png)
# 方法
## Swin-Transformer
与 ResNet 相比，Swin-Transformer 能够提供更紧凑的特征表示和更丰富得到语义信息。

与 Siamese 网络一致，SwinTrack 需要Template与search两个输入，两个输入都如 Swin-Transformer 的处理方式一样，被分割为不重叠的 patch 送入网络，分别得到 $\varphi(z)$ 与 $\varphi(x)$。

## 视觉-运动特征编码器-解码器
### 编码器
将template特征 $\varphi(z)$ 与search特征 $\varphi(x)$ 简单的拼接得到混合表示 $f^1_m$ 输入编码器融合。编码器是 Transformer 编码器。
### 解码器
解码器需要一个通过目标对象历史轨迹生成的motion token，历史轨迹被表示为一组目标对象边界框 $\mathcal{T} = \{o_{s(1)},...,o_{s(n)}\}$，$s(n)$ 表示从当前时刻开始每间隔一段时间采样到的坐标，$t$ 时刻的边界框 $\mathbb{o}_t=\{o^{x_1}_t,o^{y_1}_t,o^{x_2}_t,o^{y_1}_t\}$，因为输入图像被进行了处理（缩放填充平移之类的），所以对轨迹应用相同的变换，使其对裁剪不变，得到 $\bar{\mathcal{T}} = \{\bar{o}_{s(t)},...,\bar{o}_{s(n)}\}$。拆分后得到输出 $f^L_z,f^L_x$

将目标对象框的 4 个坐标被归一化后乘以 $g$ 取整为 $[1,g]$ 范围内的整数，如果不存在对象则设为 $g+1$。

$$ n(o,l)=\left\{
\begin{aligned}
&\lfloor \frac{o}{l} \times g \rfloor &&\text{如果对象存在} \\
&g+1,&&\text{else}
\end{aligned}
\right .$$

$$ \hat{o}_t = \{n(\bar{o}^{x_1}_t,w),n(\bar{o}^{y_1}_t,h),n(\bar{o}^{x_2}_t,w),n(\bar{o}^{y_2}_t,h)\} \in \mathbb{R}^{1 \times 4}$$
最终 motion token为所有对象框拼接而成：
$$ E_{motion} = concat(\hat{o}_{s(t)},...,\hat{o}_{s(n)}) \in \mathbb{R}^{1 \times d}$$

解码器并不是Transformer的自回归解码器，而是简单的将motion token与编码器的输出拼接得到
$$ f^D_m = concat(E_{motion},f^L_z,f^L_x) $$
经过多头交叉注意力：
$$ f^\prime_{vm} = f^L_x + MCA(LN(f^L_x),LN(f^D_m)) $$
前馈神经网络：
$$ f_{vm} = f^\prime_{vm} + FFN(LN(f^\prime_{vm})) $$

### 头部网络
分为两个分支：分类和边界框回归，每一个分支都是一个三层感知器。分别预测分类响应图 $r_{cls} \in \mathbb{R}^{(H_x \times W_x) \times 1}$ 和边界框回归映射 $r_{reg} \in \mathbb{R}^{(H_x \times W_x) \times 4}$

{% btn 'https://arxiv.org/abs/2112.00995v3',Arxiv,fa fa-file-pdf,blue %}
{% btn 'https://github.com/litinglin/swintrack',Github,fa fa-code,red %}