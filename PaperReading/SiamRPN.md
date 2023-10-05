---
title: SiamRPN
cover: structure.png
tags:
  - Object Tracking
  - Computer Vision
categories:
  - Paper Reading
abbrlink: 48598
date: 2022-11-08 17:14:16
---
# 概述
- 将目标检测中的RPN网络应用在目标追踪。
- 两部分：
    - 用于特征提取的Siamese网络（使用预训练的AlexNet）。
    - 预测边缘框和置信度的 RPN 网络。
# 方法
![](structure.png)
## Siamese 网络
共享参数的 AlexNet：
- 第一帧给定的 $127 \times 127$ 模板图像，输入Siamese网络，得到 $256 \times 6 \times 6$ 的特征图，文中记为 $\varphi(z)$。
- 当前帧 $255 \times 255$ 的Search Image，输入Siamese网络，得到 $256 \times 22 \times 22$ 的特征图，$\varphi(x)$。
## RPN 网络
$k$ 为每个位置上生成的锚框数量。
### 分类分支
1. 将Siamese的输出 $\varphi(z)$ 和 $\varphi(x)$ 分别通过 $3 \times 3$ 卷积层（两个卷积参数不共享）映射到 $4 \times 4 \times (2k \times 256)$ （$2k$：$(positive,negative)$）和 $20 \times 20 \times 256$，即为 $[\varphi(z)]_{cls}$ 和 $[\varphi(x)]_{cls}$
2. $[\varphi(z)]_{cls}$ 以“组”的方式作为 $[\varphi(x)]_{cls}$ 的卷积核，也就是说，$[\varphi(z)]_{cls}$ 一组中的通道数与 $[\varphi(x)]_{cls}$ 整体的通道数相同，即 $[\varphi(z)]_{cls}$ 通道数为 $2k$ 为 $256$ 的卷积核，分别将这 $2k$ 组卷积核与 $[\varphi(x)]_{cls}$ 卷积得到 $2k$ 组通道为 $1$ 的特征图，然后沿着通道拼接最终得到一个通道数为 $2k$ 的特征图 $A^{cls}_{17 \times 17 \times 2k} = [\varphi(x)]_{cls} * [\varphi(z)]_{cls}$，表示 $[x^{cls}_i,y^{cls}_j,c^{cls}_l],i \in [0,w),j=[0,h),l=[0,2k)$。
### 回归分支
类似的：
1. 通过卷积映射到 $4 \times 4 (\times 4k \times 256)$ （$4k$：$(x,y,w,h)$）和 $20 \times 20 \times 256$ 得到 $[\varphi(z)]_{reg}$ 和 $[\varphi(x)]_{reg}$。
2. $A^{reg}_{17 \times 17 \times 4k} = [\varphi(x)]_{reg} * [\varphi(z)]_{reg}$，表示 $[x^{reg}_i,y^{reg}_j,[dx^{reg}_p,dy^{reg}_p,dw^{reg}_p,dh^{reg}_p]],i \in [0,w),j=[0,h),p=[0,k)$
# 损失函数
## 回归损失
回归分支的输出 $A_x,A_y,A_w,A_h$，Ground Truth $T_x,T_y,T_w,T_h$ 两者距离为：
$$ (T_x-A_x),(T_y-A_y),(\frac{T_w}{A_w}),(\frac{T_h}{A_h}) $$

为了消除不同大小锚框的尺寸差异，引入正则化后的 $\delta$。
$$ \delta[0]= \frac{T_x-A_x}{A_w},\delta[1]= \frac{T_y-A_y}{A_h} \\ \delta[2]= \ln{\frac{T_w}{A_w}},\delta[3]= \ln{\frac{T_h}{A_h}}$$

然后通过 Smooth $L_1$ 损失：
$$ smooth_{L_1}(x,\sigma)=\left\{
\begin{aligned}
&0.5\sigma^2x^2, &&|x| < \frac{1}{\sigma^2} \\
&|x| - \frac{1}{2\sigma^2},&&else
\end{aligned}
\right .$$
回归损失被写为：
$$ L_{reg} = \sum^3_{i=0} smooth_{L_1}(\delta[i],\sigma) $$
## 分类损失
交叉熵损失
## 总体损失
$$ loss = L_{cls} + \lambda L_{reg} $$

# Anchor 设置
![](proposal.png)
只对特征中间小一圈的范围内每个点生成长宽比为 $(\frac{1}{3}, \frac{1}{2}, 1, 2, 3)$ 的5个锚框。因为上一帧检测到的锚框被变换到了图像中间（输入图像的处理与SiamFC的处理方法一致），而这一帧与上一帧的位置不会变化太大，即也在中间的小范围内。

{% btn 'https://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html',CVPR,fa fa-file-pdf,blue %}
{% btn 'https://github.com/foolwood/DaSiamRPN',Github,fa fa-code,red %}