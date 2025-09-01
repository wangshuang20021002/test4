# Test-Agnostic Long-Tailed Recognition  

This repository is the official Pytorch implementation of [Breaking Long-Tailed Learning Bottlenecks: A Controllable Paradigm with Hypernetwork-Generated Diverse Experts](https://openreview.net/pdf?id=WpPNVPAEyv) (NeurIPS 2024).

## 1. Requirements
* To install requirements: 
```
pip install -r requirements.txt
```

## 2. Datasets
### (1) Four bechmark datasets 
* Please download these datasets and put them to the /data file.
* ImageNet-LT and Places-LT can be found at [here](https://drive.google.com/drive/u/1/folders/1j7Nkfe6ZhzKFXePHdsseeeGI877Xu1yf).
* iNaturalist data should be the 2018 version from [here](https://github.com/visipedia/inat_comp).
* CIFAR-100 will be downloaded automatically with the dataloader.

```
data
├── ImageNet_LT
│   ├── test
│   ├── train
│   └── val
├── CIFAR100
│   └── cifar-100-python
├── Place365
│   ├── data_256
│   ├── test_256
│   └── val_256
└── iNaturalist 
    ├── test2018
    └── train_val2018
```

### (2) Txt files
* We provide txt files for test-agnostic long-tailed recognition for ImageNet-LT, Places-LT and iNaturalist 2018. CIFAR-100 will be generated automatically with the code.
* For iNaturalist 2018, please unzip the iNaturalist_train.zip.
```
data_txt
├── ImageNet_LT
│   ├── ImageNet_LT_backward2.txt
│   ├── ImageNet_LT_backward5.txt
│   ├── ImageNet_LT_backward10.txt
│   ├── ImageNet_LT_backward25.txt
│   ├── ImageNet_LT_backward50.txt
│   ├── ImageNet_LT_forward2.txt
│   ├── ImageNet_LT_forward5.txt
│   ├── ImageNet_LT_forward10.txt
│   ├── ImageNet_LT_forward25.txt
│   ├── ImageNet_LT_forward50.txt
│   ├── ImageNet_LT_test.txt
│   ├── ImageNet_LT_train.txt
│   ├── ImageNet_LT_uniform.txt
│   └── ImageNet_LT_val.txt
├── Places_LT_v2
│   ├── Places_LT_backward2.txt
│   ├── Places_LT_backward5.txt
│   ├── Places_LT_backward10.txt
│   ├── Places_LT_backward25.txt
│   ├── Places_LT_backward50.txt
│   ├── Places_LT_forward2.txt
│   ├── Places_LT_forward5.txt
│   ├── Places_LT_forward10.txt
│   ├── Places_LT_forward25.txt
│   ├── Places_LT_forward50.txt
│   ├── Places_LT_test.txt
│   ├── Places_LT_train.txt
│   ├── Places_LT_uniform.txt
│   └── Places_LT_val.txt
└── iNaturalist18
    ├── iNaturalist18_backward2.txt
    ├── iNaturalist18_backward3.txt
    ├── iNaturalist18_forward2.txt
    ├── iNaturalist18_forward3.txt
    ├── iNaturalist18_train.txt
    ├── iNaturalist18_uniform.txt
    └── iNaturalist18_val.txt 
```


## 3. Script

### (1) ImageNet-LT
#### Training
* To train the expertise-diverse model, run this command:
```
python train.py -c configs/config_imagenet_lt_resnext50_sade.json
```

#### Evaluate
* To evaluate expertise-diverse model on the uniform test class distribution, run:
``` 
python test.py -r checkpoint_path
``` 

* To evaluate expertise-diverse model on agnostic test class distributions, run:
``` 
python test_all_imagenet.py -r checkpoint_path
``` 

#### Test-time training
* To test-time train the expertise-diverse model for agnostic test class distributions, run:
``` 
python test_training_imagenet.py -c configs/test_time_imagenet_lt_resnext50_sade.json -r checkpoint_path
``` 



### (2) CIFAR100-LT 
#### Training
* To train the expertise-diverse model, run this command:
```
python train.py -c configs/config_cifar100_ir100_sade.json
```
* One can change the imbalance ratio from 100 to 10/50 by changing the config file.

#### Evaluate
* To evaluate expertise-diverse model on the uniform test class distribution, run:
``` 
python test.py -r checkpoint_path
``` 

* To evaluate expertise-diverse model on agnostic test class distributions, run:
``` 
python test_all_cifar.py -r checkpoint_path
``` 

#### Test-time training
* To test-time train the expertise-diverse model for agnostic test class distributions, run:
``` 
python test_training_cifar.py -c configs/test_time_cifar100_ir100_sade.json -r checkpoint_path
``` 
* One can change the imbalance ratio from 100 to 10/50 by changing the config file.
 

### (3) Places-LT
#### Training
* To train the expertise-diverse model, run this command:
```
python train.py -c configs/config_places_lt_resnet152_sade.json
```

#### Evaluate
* To evaluate expertise-diverse model on the uniform test class distribution, run:
``` 
python test_places.py -r checkpoint_path
``` 

* To evaluate expertise-diverse model on agnostic test class distributions, run:
``` 
python test_all_places.py -r checkpoint_path
``` 

#### Test-time training
* To test-time train the expertise-diverse model for agnostic test class distributions, run:
``` 
python test_training_places.py -c configs/test_time_places_lt_resnet152_sade.json -r checkpoint_path
``` 

### (4) iNaturalist 2018
#### Training
* To train the expertise-diverse model, run this command:
```
python train.py -c configs/config_iNaturalist_resnet50_sade.json
```

#### Evaluate
* To evaluate expertise-diverse model on the uniform test class distribution, run:
``` 
python test.py -r checkpoint_path
``` 

* To evaluate expertise-diverse model on agnostic test class distributions, run:
``` 
python test_all_inat.py -r checkpoint_path
``` 

#### Test-time training
* To test-time train the expertise-diverse model for agnostic test class distributions, run:
``` 
python test_training_inat.py -c configs/test_time_iNaturalist_resnet50_sade.json -r checkpoint_path
``` 

## 4. Citation
If you find our work inspiring or use our codebase in your research, please cite our work.
```
@article{zhao2024breaking,
  title={Breaking Long-Tailed Learning Bottlenecks: A Controllable Paradigm with Hypernetwork-Generated Diverse Experts},
  author={Zhao, Zhe and Wen, HaiBin and Wang, Zikang and Wang, Pengkun and Wang, Fanfu and Lai, Song and Zhang, Qingfu and Wang, Yang},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={7493--7520},
  year={2024}
}
``` 

## 5. Acknowledgements
This is a project based on this [pytorch template](https://github.com/victoresque/pytorch-template). 

The mutli-expert framework are based on [RIDE](https://github.com/frank-xwang/RIDE-LongTailRecognition). The data generation of agnostic test class distributions takes references from [LADE](https://github.com/hyperconnect/LADE).



# Create the Markdown file with the frequency-domain (real-part only) NMR denoising plan
content = r"""# 频域（仅实部）NMR 去噪方案

> 适用前提：同一样本存在多次平行实验，实验 \(k=0,1,\dots,K\) 的**信噪比（SNR）按固定倍数 \(T>1\)** 逐步提升。每条谱是**频域实部**（real-valued 1D）。所有谱在采集后均做过**归一化**（但归一化方式可能不同）。目标：学习去噪映射 \(f_\theta\)，使其在原始低 SNR 输入上输出更接近真实光谱、保峰保线型、稳基线。

---

## 1. 数据建模与记号

- 第 \(k\) 次实验的未归一化频域实部记为 \(\tilde{x}^{(k)}\in\mathbb{R}^L\)（长度 \(L\)）：  
  $$
  \tilde{x}^{(k)} \;=\; a_k\, s \;+\; b_k\, n^{(k)},\qquad k=0,\dots,K,
  $$
  其中 \(s\) 为理想真实光谱（同一样本不变），\(n^{(k)}\) 为零均值噪声；相邻实验的 SNR 满足  
  $$
  \frac{a_{k+1}/b_{k+1}}{a_k/b_k} \;\approx\; T\;>\;1.
  $$

- 实际可用数据为归一化后的 \(x^{(k)}=\mathrm{norm}(\tilde{x}^{(k)})\)。由于归一化可能改变幅度尺度，后续我们会在**峰区**做**鲁棒幅度重标定**以便可比。

- 训练目标：学习 \(f_\theta:\mathbb{R}^L\!\to\!\mathbb{R}^L\)，令 \(f_\theta(x^{(0)})\) 在尺度可比的意义下逼近 \(s\)。

> 若只有频域幅度谱（实部），直接在频域做 1D 去噪；若之后拥有时域 FID，可扩展为双域训练。

---

## 2. 频域预处理流程（仅实部）

### 2.1 频轴对齐（ppm shift 校正）
对同一样本的 \(\{x^{(k)}\}\) 做亚像素对齐，避免峰位漂移影响融合与损失：
- 使用**相位互相关**或**互相关峰定位**求相对位移 \(\Delta \tau_k\)，对谱作循环平移或线性插值校正；
- 如存在局部非线性漂移，可在若干**锚峰窗口**内做**局部对齐**（优先对强峰）。

### 2.2 鲁棒幅度重标定（跨 \(k\) 使峰区可比）
归一化破坏了原幅度比例。设 \(x^{(K)}\) 为最高 SNR 谱（或 §3.1 的加权融合伪真值）。在“候选峰区”掩码 \(M\in\{0,1\}^L\) 内，用最小二乘求每条谱的**标定系数** \(s_k\)：  
$$
s_k \;=\; \arg\min_{s>0}\; \sum_{i=1}^{L} M_i \,\big(s\,x^{(k)}_i - x^{(K)}_i\big)^2
\;=\;
\frac{\sum_i M_i\, x^{(K)}_i\, x^{(k)}_i}{\sum_i M_i\, \big(x^{(k)}_i\big)^2}.
$$
重标定后 \(\bar{x}^{(k)}=s_k\,x^{(k)}\)。

> 若不想以 \(x^{(K)}\) 作参照，可先做一次“融合伪真值”（见 §3.1）再回到此步重标定。

### 2.3 峰区/基线区自动分割（峰掩码 \(M\) 与基线掩码 \(B\)）
1. **平滑**：如 Savitzky–Golay 得到 \(x^{(k)}_{\text{smooth}}\)；  
2. **候选峰检测**：对 \(\{\bar{x}^{(k)}\}\) 取逐点最大 \(\max_k \bar{x}^{(k)}\)；  
3. 计算自适应阈值 \(\tau = \mathrm{median} + \alpha \cdot \mathrm{MAD}\)（\(\alpha\in[3,5]\)）；  
4. 令 \(M_i=\mathbf{1}\{\max_k \bar{x}^{(k)}_i\ge \tau\}\)，再**形态学膨胀**若干点以覆盖峰裙；  
5. 基线掩码 \(B\) 取 \(M\) 的补集，并**腐蚀** 1–2 点避免峰脚漏入。

### 2.4 噪声强度与 SNR 估计（在 \(B\) 内）
基线噪声标准差（鲁棒）：  
$$
\sigma_k \;=\; 1.4826 \cdot \mathrm{median}\Big(\,\big|\bar{x}^{(k)}_i - \mathrm{median}(\bar{x}^{(k)}_{B})\big| \;:\; i\in B\Big).
$$
稳健 SNR（用于权重）：  
$$
\mathrm{SNR}_k \;=\; \frac{\mathrm{P95}\big(\bar{x}^{(k)}_{M}\big) - \mathrm{median}\big(\bar{x}^{(k)}_{B}\big)}{\sigma_k}.
$$

> 可据 \(\sigma_k\) **自适应估计** \(T\)：\(\widehat{T}\approx \mathrm{median}_k(\sigma_k/\sigma_{k+1})\)。

---

## 3. 监督目标（频域实部）

### 3.1 伪真值（方案 S）：多 SNR 融合
用 \(\{\bar{x}^{(k)}\}\) 在频域做加权得到伪真值 \(\hat{s}\)：  
$$
\hat{s}_i \;=\;
\frac{\sum_{k=0}^{K} w_k \,\bar{x}^{(k)}_i}{\sum_{k=0}^{K} w_k},
\qquad
w_k \;\propto\; \mathrm{SNR}_k^{\,\gamma}\;\; \text{或}\;\; T^{k},\;\; \gamma\in[0.5,1.5].
$$
为抑制残噪放大，建议**上限截断**：\(w_k\leftarrow \min(w_k, w_{\max})\)。

> 简化选项：若最高 SNR 谱质量已足够，可直接取 \(\hat{s}=\bar{x}^{(K)}\)。

### 3.2 自监督一致性（方案 U）：多 SNR 对齐
对同一样本的多 SNR 谱经模型输出后，在频域直接对齐：  
$$
\mathcal{L}_{\mathrm{cons}} \;=\;
\frac{1}{\binom{K+1}{2}}
\sum_{0\le i<j\le K}
\Big\|\, f_\theta(\bar{x}^{(i)}) - f_\theta(\bar{x}^{(j)}) \Big\|_{1}.
$$

### 3.3 残差比例软约束（用 \(T\) 的先验）
定义残差 \(r^{(k)}=\bar{x}^{(k)}-f_\theta(\bar{x}^{(k)})\)。在**基线区 \(B\)** 控制能量比：  
$$
\mathcal{R}_{ij} \;=\; 
\frac{\| r^{(i)}\odot B \|_2}{\| r^{(j)}\odot B \|_2 + \varepsilon},
\qquad
\alpha_{ij} \;\approx\; T^{-(j-i)}.
$$
加入软约束损失（取对数可稳定）：  
$$
\mathcal{L}_{T} \;=\;
\frac{1}{\binom{K+1}{2}}
\sum_{i<j} \Big| \log \mathcal{R}_{ij} - \log \alpha_{ij} \Big|.
$$

---

## 4. 模型结构（频域 1D，实部）

### 4.1 残差式 1D U-Net（推荐）
- **输入/输出**：\(\mathbb{R}^{L}\to\mathbb{R}^{L}\)。输出表示**去噪后的谱**（也可输出噪声残差，推理时输入减残差）。  
- **编码器（4 层）**：每层 Conv1D(k=9, dilation=1/2/4/8) + Norm(Group/LayerNorm) + GELU + 残差，stride 2 下采样。  
- **解码器（4 层）**：转置卷积上采样 + 跳连（concat 编码器同级特征）+ 残差块。  
- **通道数**：64→128→256→256（瓶颈）→…→64。  
- **注意力（可选）**：SE/CBAM 1D 放在瓶颈与浅层以增强峰区表达。  
- **输出头**：Conv1D(k=1)；线性或 `tanh` 后与输入做短路残差。

> 1D 序列通常对轻量模型友好；总参数量 1–3M 足够。

---

## 5. 频域损失设计（仅实部）

令 \(y^{(k)} = f_\theta(\bar{x}^{(k)})\)。

### 5.1 重建与线型保持（对伪真值）
- **重建损失**（对目标样本 \(k=0\) 必选；其余样本可采样使用）：  
  $$
  \mathcal{L}_{\mathrm{rec}} \;=\;
  \big\|\, y^{(0)} - \hat{s} \,\big\|_{1}.
  $$
- **一阶导数损失**（保峰形）：  
  $$
  \mathcal{L}_{\mathrm{deriv}} \;=\;
  \big\|\nabla y^{(0)} - \nabla \hat{s}\big\|_{1},\quad
  \nabla y_i = y_{i+1}-y_i.
  $$
- **峰区权重**：对峰掩码 \(M\) 内权重放大，形成加权 \(L_1\)（如峰区 \(\times 6\)，基线 \(\times 1\)）。

### 5.2 基线稳健与平滑
- **二阶差分正则（仅在基线区）**：  
  $$
  \mathcal{L}_{\mathrm{curv}} \;=\;
  \big\|\Delta y^{(0)}\odot B\big\|_{1},\quad
  \Delta y_i = y_{i+1}-2y_i+y_{i-1}.
  $$

### 5.3 多 SNR 一致性与 \(T\) 约束
- **一致性损失**：\(\mathcal{L}_{\mathrm{cons}}\)（§3.2），可在所有 \(k\) 上采样计算。  
- **\(T\) 软约束**：\(\mathcal{L}_{T}\)（§3.3），仅在 \(B\) 内基于能量比。

### 5.4 总损失（示例权重）
$$
\mathcal{L}
=
\underbrace{\mathcal{L}_{\mathrm{rec}} + \lambda_1 \mathcal{L}_{\mathrm{deriv}} + \lambda_2 \mathcal{L}_{\mathrm{curv}}}_{\text{监督（对 }k=0\text{ 与伪真值）}}
\;+\;
\underbrace{\lambda_3 \mathcal{L}_{\mathrm{cons}} + \lambda_4 \mathcal{L}_{T}}_{\text{自监督（跨 }k\text{ 一致与 }T\text{ 约束）}}.
$$

建议初始权重：\(\lambda_1=0.5,\; \lambda_2=0.1,\; \lambda_3=0.5,\; \lambda_4=0.2\)。

---

## 6. 训练策略（两阶段）

### 阶段 A（监督/弱监督为主，30–50 epoch）
- 以 \(\hat{s}\)（§3.1）作为伪真值，主训练 \(\mathcal{L}_{\mathrm{rec}}+\lambda_1\mathcal{L}_{\mathrm{deriv}}+\lambda_2\mathcal{L}_{\mathrm{curv}}\)。  
- 批次中除 \(k=0\) 外，随机采样 1–2 条 \(k>0\) 谱参与 \(\mathcal{L}_{\mathrm{cons}}\)。

### 阶段 B（自监督增强，40–70 epoch）
- 加大 \(\lambda_3\)（一致性）与 \(\lambda_4\)（\(T\) 软约束），充分利用全部 \(k\)。  
- **SNR 课程学习**：先用高 SNR 对（大 \(k\)），再逐步加入低 SNR 对（小 \(k\)）。  
- 训练后期可上调峰区权重，或下调 \(\lambda_2\) 防止过抹峰。

**优化器与调度（建议）**：AdamW（lr=3e-4，wd=1e-4），cosine decay，warmup 5 epoch。  
**批次构成**：同一 batch 内尽量包含**同一样本的多 SNR**（利于一致性计算）。

---

## 7. 评测指标（频域，仅实部）

1. **SNR 提升**（统一窗口）：  
   \(\mathrm{SNR}_{\mathrm{out}} = \dfrac{\mathrm{P95}(y_M)-\mathrm{median}(y_B)}{\mathrm{std}(y_B)}\)，与输入/伪真值对比。
2. **峰检测/定位**：对 \(y^{(0)}\) 与 \(\hat{s}\) 做峰匹配（容忍 \(\pm\Delta\) ppm），统计 Precision/Recall/F1、定位误差。
3. **线型量化**：主峰 **FWHM** 与**积分面积**相对误差。
4. **整体失真**：\(\|y^{(0)}-\hat{s}\|_1,\;\|y^{(0)}-\hat{s}\|_2\)。

---

## 8. 消融建议

- **峰区权重**：\(+/-\) 峰掩码加权。  
- **\(\mathcal{L}_{\mathrm{curv}}\)**：基线稳定性对比。  
- **一致性/ \(T\) 约束**：去掉 \(\mathcal{L}_{\mathrm{cons}}\)、\(\mathcal{L}_T\) 的影响。  
- **伪真值策略**：仅 \(k=K\) vs SNR 加权融合。  
- **重标定**：有/无 §2.2 的 \(s_k\) 标定对效果的影响。

---

## 9. 训练主循环（频域伪代码，PyTorch 风格）

```python
# x_list: [B, K+1, L]  —— 已对齐与重标定（§2.1/2.2）；
# mask_peak, mask_base: [B, L]；s_hat: [B, L]（§3.1）

for epoch in range(num_epochs):
    for x_list, mask_peak, mask_base, s_hat in loader:
        x0 = x_list[:, 0, :]          # 低SNR输入
        y0 = model(x0)                # 去噪输出

        # 峰/基线加权
        w_peak, w_base = 6.0, 1.0
        W = w_base*mask_base + w_peak*mask_peak

        # 重建 + 一阶导数
        L_rec   = (W * (y0 - s_hat).abs()).mean()

        def diff1(z): return z[..., 1:] - z[..., :-1]
        L_deriv = (diff1(y0) - diff1(s_hat)).abs().mean()

        # 基线二阶差分（仅在B内）
        def diff2(z): return z[..., 2:] - 2*z[..., 1:-1] + z[..., :-2]
        L_curv  = (diff2(y0) * mask_base[..., 1:-1]).abs().mean()

        # 多SNR一致性（随机采 1-2 条高SNR作为对照）
        y_all = []
        idxs = sample_indices_from_1_to_K(x_list.shape[1])
        for j in idxs:
            y_all.append(model(x_list[:, j, :]))
        L_cons = 0.0
        if len(y_all) >= 2:
            cnt = 0
            for i in range(len(y_all)):
                for j in range(i+1, len(y_all)):
                    L_cons += (y_all[i] - y_all[j]).abs().mean()
                    cnt += 1
            L_cons /= max(1, cnt)

        # 残差比例软约束（在基线区）
        eps = 1e-8
        L_T = 0.0; cnt = 0
        for i in range(len(y_all)):
            for j in range(i+1, len(y_all)):
                ri = (x_list[:, i+1, :] - y_all[i]) * mask_base
                rj = (x_list[:, j+1, :] - y_all[j]) * mask_base
                num = (ri.pow(2).sum(dim=-1).sqrt() + eps)
                den = (rj.pow(2).sum(dim=-1).sqrt() + eps)
                ratio = num / den
                alpha = (T ** (-(j - i))) * torch.ones_like(ratio)
                L_T += ( (ratio+eps).log() - (alpha+eps).log() ).abs().mean()
                cnt += 1
        L_T /= max(1, cnt)





# 3.模型介绍

# 3.1 模型设定

本文主要研究三个子组的分组情况，以 $\delta _ { i }$ 表示子组个数且 $\delta _ { i } \in \{ 1 , 2 , 3 \}$ 。随机效应 $b _ { i }$ 的协方差矩阵为 $\sigma _ { b } ^ { 2 }$ ， $\delta _ { i } = 1$ 对应随机效应 $b _ { i }$ 的均值为 $\mu _ { 1 }$ ， $\delta _ { i } = 2$ 对应随机效应 $b _ { i }$ 的均值为 $\mu _ { 2 }$ ，当$\delta _ { i } = 3$ 对应随机效应 $b _ { i }$ 的均值为 $\mu _ { 3 }$ 。此时对应子组的概率如下：

$$
P ( \delta _ { i } = k | U _ { i } ) = \frac { e ^ { U _ { i } \gamma _ { k } } } { \sum _ { k = 1 } ^ { 3 } e ^ { U _ { i } \gamma _ { k } } } , k = 1 , 2 , 3
$$

各子组的概率分别如下：

$$
\begin{array} { l } { P ( \delta _ { i } = 1 | U _ { i } ) = \displaystyle \frac { e ^ { U _ { i } \gamma _ { 1 } } } { e ^ { U _ { i } \gamma _ { 1 } } + e ^ { U _ { i } \gamma _ { 2 } } + e ^ { U _ { i } \gamma _ { 3 } } } } \\ { P ( \delta _ { i } = 2 | U _ { i } ) = \displaystyle \frac { e ^ { U _ { i } \gamma _ { 2 } } } { e ^ { U _ { i } \gamma _ { 1 } } + e ^ { U _ { i } \gamma _ { 2 } } + e ^ { U _ { i } \gamma _ { 3 } } } } \\ { P ( \delta _ { i } = 3 | U _ { i } ) = \displaystyle \frac { e ^ { U _ { i } \gamma _ { 3 } } } { e ^ { U _ { i } \gamma _ { 1 } } + e ^ { U _ { i } \gamma _ { 2 } } + e ^ { U _ { i } \gamma _ { 3 } } } } \end{array}
$$

# 3.1.4 潜在子组条件模型

为简化分析过程，本文只考虑个体间异质性造成的随机效应，不针对个体内异质性进一步分析。引入潜在子组 $\delta _ { i }$ ，考虑三个子组的情况即 $| \delta _ { i } \in \{ 1 , 2 , 3 \}$ ，潜在子组可以通过与分组相关的协变量 $U _ { i }$ 的逻辑回归构建模型，并将随机效应看作是来自三个均值不同，方差相同的正态分布即 $b _ { i } { \sim } N ( \mu _ { k } , \sigma _ { b } ^ { 2 } )$ ，则随机效应的有限混合分布：

$$
g ( b ) = \sum _ { k = 1 } ^ { 3 } p _ { k } g _ { k } ( b )
$$

其中， $\begin{array} { r } { g _ { k } ( b ) = \frac { 1 } { \sqrt { 2 \pi \sigma _ { b } ^ { 2 } } } e x p \left[ - \frac { ( b - \mu _ { k } ) ^ { 2 } } { 2 \sigma _ { b } ^ { 2 } } \right] \circ } \end{array}$ 与混合子组相对应的比例为常数 ${ \cdot } p _ { k } \ge 0 ( k = 1 , 2 , 3 )$ ，且满足 $\begin{array} { r } { \sum _ { k = 1 } ^ { 3 } p _ { k } = 1 } \end{array}$ 。引入潜在子组 $\delta _ { i }$ 的条件模型矩阵形式如下：

$$
y _ { i } | ( \delta _ { i } , U _ { i } , W _ { i } , b _ { i } ) = \left( \begin{array} { c c c c } { W _ { i 1 1 } } & { W _ { i 1 2 } } & { \cdots } & { W _ { i 1 p } } \\ { W _ { i 2 1 } } & { W _ { i 2 2 } } & { \cdots } & { W _ { i 2 p } } \\ { \vdots } & { \vdots } & { \vdots } \\ { W _ { i t 1 } } & { W _ { i t 2 } } & { \cdots } & { W _ { i t p } } \end{array} \right) \left( \begin{array} { c } { \alpha _ { 1 } } \\ { \alpha _ { 2 } } \\ { \vdots } \\ { \alpha _ { p } } \end{array} \right) + \left( \begin{array} { c } { 1 } \\ { 1 } \\ { \vdots } \\ { 1 } \end{array} \right) b _ { i } + \left( \begin{array} { c } { \varepsilon _ { i 1 } } \\ { \varepsilon _ { i 2 } } \\ { \vdots } \\ { \varepsilon _ { i t } } \end{array} \right)
$$

令 $W _ { i } = \left( \begin{array} { c c c c } { W _ { i 1 1 } } & { W _ { i 1 2 } } & { \cdots } & { W _ { i 1 p } } \\ { W _ { i 2 1 } } & { W _ { i 2 2 } } & { \cdots } & { W _ { i 2 p } } \\ { \vdots } & { \vdots } & { } & { \vdots } \\ { W _ { i t 1 } } & { W _ { i t 2 } } & { \cdots } & { W _ { i t p } } \end{array} \right)$ 表示固定效应的设计矩阵， $W _ { i } \in R ^ { t \times p }$ ； $\alpha =$ $( \alpha _ { 1 } , \alpha _ { 2 } , \cdots , \alpha _ { q } ) ^ { T }$ 表示固定效应， $\alpha \in R ^ { p \times 1 }$ ；随机效应 $b _ { i } { \sim } N ( \mu _ { k } , \sigma _ { b } ^ { 2 } )$ ； $\varepsilon _ { i t }$ 表示随机误差且$\varepsilon _ { i t } { \sim } N ( 0 , \sigma _ { \varepsilon } ^ { 2 } )$ ，令 $\mathbf { \Psi } ^ { \cdot } \varepsilon _ { i } = ( \varepsilon _ { i 1 } , \varepsilon _ { i 2 } , \cdots , \varepsilon _ { i t } ) ^ { T }$ ，则 $\varepsilon _ { i } { \sim } N ( 0 , \sigma _ { \varepsilon } ^ { 2 } R )$ 。 $\mathbf { 1 } = ( 1 , 1 , \cdots , 1 ) ^ { T }$ 表示元素全为1 的$t$ 维列向量， $R _ { t \times t }$ 表示 $\mathbf { \boldsymbol { t } } \times \mathbf { \boldsymbol { t } }$ 维工作相关矩阵。本文仅考虑平衡数据的情况，此时的工作相关矩阵为对称矩阵即 $R ^ { T } = R$ ，且 $( R ^ { - 1 } ) ^ { T } = R ^ { - 1 }$ 。随机效应与随机误差不存在相关性，彼此之间相互独立，此时潜在子组条件模型一般形式如下：

$$
y _ { i } | ( \delta _ { i } , U _ { i } , W _ { i } , b _ { i } ) = W _ { i } \alpha + { \bf 1 } b _ { i } + \varepsilon _ { i } , i = 1 , 2 , \dots , N
$$

在 $\delta _ { i } , U _ { i } , W _ { i }$ 的条件下， $y _ { i }$ 和 $b _ { i }$ 联合正态分布如下：

$$
\binom { y _ { i } } { b _ { i } } \sim N \left( \left[ \begin{array} { c c } { W _ { i } \alpha + \mathbf { 1 } \mu _ { i } } \\ { \mu _ { i } } \end{array} \right] , \left[ \begin{array} { c c } { \Sigma } & { B } \\ { B ^ { T } } & { \sigma _ { b } ^ { 2 } } \end{array} \right] _ { ( t + 1 ) \times ( t + 1 ) } \right)
$$

$$
\begin{array} { c } { \mu _ { i } = \mu _ { 1 } I ( \delta _ { i } = 1 ) + \mu _ { 2 } I ( \delta _ { i } = 2 ) + \mu _ { 3 } I ( \delta _ { i } = 3 ) } \\ { V a r ( y _ { i } ) = \Sigma = \mathbf { 1 } \sigma _ { b } ^ { 2 } \mathbf { 1 } ^ { T } + \sigma _ { \varepsilon } ^ { 2 } R _ { t \times t } } \\ { B = \mathbf { 1 } \sigma _ { b } ^ { 2 } } \\ { B ^ { T } = \sigma _ { b } ^ { 2 } \mathbf { 1 } ^ { T } } \end{array}
$$

根据高斯联合分布的条件期望与条件方差公式，给定 $U _ { i } , W _ { i }$ 且在 $y _ { i }$ 和 $\delta _ { i }$ 的条件下随机效应 $b _ { i }$ 的后验均值为

$$
\hat { b } _ { i } = E ( b _ { i } | y _ { i } , \delta _ { i } ) = \mu _ { i } + B ^ { T } \Sigma ^ { - 1 } ( y _ { i } - W _ { i } \alpha - \mathbf { 1 } \mu _ { i } )
$$

对于每个子组 $k = 1 , 2 , 3$ 时，随机效应 $b _ { i k }$ 的后验均值为

$$
\hat { b } _ { i k } = E ( b _ { i } | y _ { i } , \delta _ { i } = k ) = \mu _ { k } + B ^ { T } \Sigma ^ { - 1 } ( y _ { i } - W _ { i } \alpha - \mathbf { 1 } \mu _ { k } )
$$

给定 $U _ { i } , W _ { i }$ 且在 $y _ { i }$ 和 $\delta _ { i }$ 的条件下随机效应 $b _ { i }$ 的后验方差为

$$
\hat { \sigma } _ { b } ^ { 2 } = C o v ( b _ { i } | y _ { i } , \delta _ { i } ) = \sigma _ { b } ^ { 2 } - B ^ { T } \Sigma ^ { - 1 } B
$$

此外，给定 $U _ { i } , W _ { i }$ 且在 $y _ { i }$ 和 $\delta _ { i }$ 的条件下随机效应的二阶矩 $b _ { i } b _ { i } ^ { T }$ 的后验估计为

$$
\boldsymbol { E } \big ( b _ { i } \boldsymbol { b } _ { i } ^ { T } \big | y _ { i } , \delta _ { i } = \boldsymbol { k } \big ) = \widehat { b } _ { i k } \widehat { \boldsymbol { b } } _ { i k } ^ { \ T } + \widehat { \sigma } _ { b } ^ { 2 }
$$

# 3.2 模型参数估计及分组

# 3.2.2 参数估计及分组

本节中我们将随机效应 $b _ { i }$ 和潜在子组变量 $\delta _ { i }$ 看作是缺失的，将待估计参数记作𝜃且 $\theta =$ $( \gamma _ { 1 } , \gamma _ { 2 } , \gamma _ { 3 } , \mu _ { 1 } , \mu _ { 2 } , \mu _ { 3 } , \alpha , \sigma _ { \varepsilon } ^ { 2 } , \sigma _ { b } ^ { 2 } , R )$ ，则完全数据的对数似然函数

$$
l ( \theta ) = \sum _ { i = 1 } ^ { N } \lbrace I ( \delta _ { i } = 1 ) l o g \pi ( U _ { i } \gamma _ { 1 } ) + I ( \delta _ { i } = 2 ) l o g \pi ( U _ { i } \gamma _ { 2 } ) + I ( \delta _ { i } = 3 ) l o g \pi ( U _ { i } \gamma _ { 3 } ) \rbrace
$$

$$
\begin{array} { l } { { + \displaystyle \sum _ { i = 1 } ^ { N } \lbrace I ( \delta _ { i } = 1 ) l o g \phi ( y _ { i } , W _ { i } \alpha + { \bf 1 } b _ { i } , \sigma _ { \varepsilon } ^ { 2 } R _ { t \times t } ) + I ( \delta _ { i } = 2 ) l o g \phi ( y _ { i } , W _ { i } \alpha + { \bf 1 } b _ { i } , \sigma _ { \varepsilon } ^ { 2 } R _ { t \times t } ) } } \\ { { + I ( \delta _ { i } = 3 ) l o g \phi ( y _ { i } , W _ { i } \alpha + { \bf 1 } b _ { i } , \sigma _ { \varepsilon } ^ { 2 } R _ { t \times t } ) \rbrace } } \\ { { + \displaystyle \sum _ { i = 1 } ^ { N } \lbrace I ( \delta _ { i } = 1 ) l o g \varphi ( b _ { i } , \mu _ { 1 } , \sigma _ { b } ^ { 2 } ) + I ( \delta _ { i } = 2 ) l o g \varphi ( b _ { i } , \mu _ { 2 } , \sigma _ { b } ^ { 2 } ) + I ( \delta _ { i } = 3 ) l o g \varphi ( b _ { i } , \mu _ { 3 } , \sigma _ { b } ^ { 2 } ) \rbrace } } \end{array}
$$

其中 $\begin{array} { r } { \pi ( x ) = \frac { e ^ { x } } { \sum _ { k = 1 } ^ { 3 } e ^ { U _ { i } \gamma _ { k } } } } \end{array}$ 表示由多项逻辑回归求得的隶属各子组的概率。 $\phi ( y _ { i } , W _ { i } \alpha +$ $\mathbf { 1 } b _ { i } , \sigma _ { \varepsilon } ^ { 2 } R _ { t \times t } )$ 表示期望为 $W _ { i } \alpha + \mathbf { 1 } b _ { i }$ ，方差为 $\sigma _ { \varepsilon } ^ { 2 } R _ { t \times t }$ ， $y _ { i }$ 的多元正态分布的密度函数。$\varphi ( b _ { i } , \mu _ { k } , \sigma _ { b } ^ { 2 } )$ 表示期望为 $\mu _ { k }$ ，方差为 $\sigma _ { b } ^ { 2 }$ ， $b _ { i }$ 的一元正态分布的密度函数。 $I ( \cdot )$ 表示指示函数，其取值为 0 或 1，当指示函数中的条件判断命题成立时，指示函数取值为 1，否则为 $0$ 。对于 $I ( \delta _ { i } = 1 )$ ， $I ( \delta _ { i } = 2 )$ ， $I ( \delta _ { i } = 3 )$ 当有一个指示函数取值为 1，则另外两个指示函数为 0。

下面通过EM 算法求完全数据对数似然函数中的未知参数。

E 步：给定当前的参数估计值 $\theta ^ { ( j ) }$ ，有 $\mathrm { Q } \big ( \theta , \theta ^ { ( j ) } \big ) = I _ { 1 } + I _ { 2 } + I _ { 3 }$ ，具体如下：

$$
I _ { 1 } = \sum _ { i = 1 } ^ { N } \{ E ( I ( \delta _ { i } = 1 ) | y ) l o g \pi ( U _ { i } \gamma _ { 1 } ) + E ( I ( \delta _ { i } = 2 ) | y ) l o g \pi ( U _ { i } \gamma _ { 2 } ) + E ( I ( \delta _ { i } = 3 ) | ) l o g \pi ( U _ { i } \gamma _ { 3 } ) \}
$$

$$
\begin{array} { l } { { I _ { 2 } = - \displaystyle \frac { M } { 2 } l o g \sigma _ { \varepsilon } ^ { 2 } - \displaystyle \frac { N } { 2 } \log | R | } } \\ { { \displaystyle + \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { 3 } - \displaystyle \frac { P ( \delta _ { i } = k | y ) } { 2 \sigma _ { \varepsilon } ^ { 2 } } ( \bigl ( y _ { i } - W _ { i } { \hat { \alpha } } - \mathbf { 1 } \hat { b } _ { i k } \bigr ) ^ { T } R ^ { - 1 } \bigl ( y _ { i } - W _ { i } { \hat { \alpha } } - \mathbf { 1 } \hat { b } _ { i k } \bigr ) + t r ( R ^ { - 1 } \mathbf { 1 } \hat { \sigma } _ { b } ^ { 2 } \mathbf { 1 } ^ { T } ) + \hat { \rho } _ { i k } \hat { \rho } _ { k } ^ { 2 } \mathbf { 1 } ^ { T } ) } } \end{array}
$$

$$
I _ { 3 } = - \frac { N } { 2 } l o g \sigma _ { b } ^ { 2 } + \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { 3 } - \frac { P ( \delta _ { i } = k | y ) } { 2 \sigma _ { b } ^ { 2 } } \Big ( \big ( \hat { b } _ { i k } - \mu _ { k } \big ) ^ { 2 } + \hat { \sigma } _ { b } ^ { 2 } \Big )
$$

其中， $\begin{array} { r } { \mathbf { M } = \sum _ { i = 1 } ^ { N } t = N t ; \hat { b } _ { i k } , \hat { \sigma } _ { b } ^ { 2 } } \end{array}$ 分别表示随机效应属于第 $k$ 子组的后验均值和后验方差。

给定当前参数 $\theta ^ { ( j ) }$ ，对于 $i = 1 , 2 , \cdots , N$ ，令

$$
P _ { i 1 } = P \big ( \delta _ { i } = 1 \big | y _ { i } , \theta ^ { ( j ) } \big )
$$

$$
\begin{array} { r l } & { = \frac { \pi ( U _ { i } \gamma _ { 1 } ^ { ( \beta ) } ) \phi ( y _ { i } , W _ { i } \alpha + \mu _ { 1 } ^ { ( \beta ) } , \Sigma _ { i } ^ { ( i ) } ) } { \pi ( U _ { i } \gamma _ { 1 } ^ { ( 1 ) } ) \phi ( y _ { i } , W _ { i } \alpha + \mu _ { 1 } ^ { ( \beta ) } , \Sigma _ { i } ^ { ( \beta ) } ) + \pi ( U _ { i } \gamma _ { 2 } ^ { ( 1 ) } ) \phi ( y _ { i } , W _ { i } \alpha + \mu _ { 2 } ^ { ( \beta ) } , \Sigma _ { i } ^ { ( i ) } ) + \pi ( U _ { i } \gamma _ { 3 } ^ { ( 0 ) } ) \phi ( y _ { i } , W _ { i } \alpha - \mu _ { 1 } ^ { ( \beta ) } ) } } \\ & { P _ { i 2 } = P ( \delta _ { i } = 2 | y _ { i } , \theta ^ { ( \beta ) } ) } \\ & { = \frac { \pi ( U _ { i } \gamma _ { 2 } ^ { ( \beta ) } ) \phi ( y _ { i } , W _ { i } \alpha + \mu _ { 1 } ^ { ( \beta ) } , \Sigma _ { i } ^ { ( \beta ) } ) } { \pi ( U _ { i } \gamma _ { 1 } ^ { ( 0 ) } ) \phi ( y _ { i } , W _ { i } \alpha + \mu _ { 2 } ^ { ( \beta ) } , \Sigma _ { i } ^ { ( i ) } ) + \pi ( U _ { i } \gamma _ { 3 } ^ { ( 0 ) } ) \phi ( y _ { i } , W _ { i } \alpha - \mu _ { 1 } ^ { ( \beta ) } ) } } \\ & { P _ { i 3 } = P ( \delta _ { i } = 3 | y _ { i } , \theta ^ { ( \beta ) } ) } \\ &  = \frac { \pi ( U _ { i } \gamma _ { 3 } ^ { ( \beta ) } ) \phi ( y _ { i } , W _ { i } \alpha + \mu _ { 1 } ^ { ( \beta ) } , \Sigma _ { i } ^ { ( \beta ) } ) + \pi ( U _ { i } \gamma _ { 3 } ^ { ( \beta ) } ) \phi ( y _ { i } , W _ { i } \alpha + \mu _ { 2 } ^ { ( \beta ) } , \Sigma _ { i } ^ { ( i ) } ) }  \pi ( U _ { i } \gamma _ { 1 } ^ { ( 1 ) } ) \phi ( y _ { i } , W _ { i } \alpha + \mu _ { 1 } ^ { ( \beta ) } , \Sigma _ { i } ^ { ( i ) } ) + \pi ( U _ { i } \gamma _ { 3 } ^ { ( \beta ) } ) \phi ( y _ { i } , W _ { i } \alpha + \mu _ { 2 } ^ { ( \beta ) } , \Sigma _ { i } ^  ( i \end{array}
$$

M 步： $\theta ^ { ( j + 1 ) } = a r g m a x _ { \theta } Q \big ( \theta , \theta ^ { ( j ) } \big )$

（1）关于子组隶属参数 $\gamma _ { 1 }$ 、 $\gamma _ { 2 }$ 、 $\gamma _ { 3 }$ 的计算：

$$
\begin{array} { r } { \gamma _ { 1 } ^ { ( j + 1 ) } = a r g m a x _ { \gamma _ { 1 } } P _ { i 1 } l o g \pi ( U _ { i } \gamma _ { 1 } ) + P _ { i 2 } l o g \pi ( U _ { i } \gamma _ { 2 } ) + P _ { i 3 } l o g \pi ( U _ { i } \gamma _ { 3 } ) } \\ { \gamma _ { 2 } ^ { ( j + 1 ) } = a r g m a x _ { \gamma _ { 2 } } P _ { i 1 } l o g \pi ( U _ { i } \gamma _ { 1 } ) + P _ { i 2 } l o g \pi ( U _ { i } \gamma _ { 2 } ) + P _ { i 3 } l o g \pi ( U _ { i } \gamma _ { 3 } ) } \\ { \gamma _ { 3 } ^ { ( j + 1 ) } = a r g m a x _ { \gamma _ { 3 } } P _ { i 1 } l o g \pi ( U _ { i } \gamma _ { 1 } ) + P _ { i 2 } l o g \pi ( U _ { i } \gamma _ { 2 } ) + P _ { i 3 } l o g \pi ( U _ { i } \gamma _ { 3 } ) } \end{array}
$$

（2）关于子组均值参数 $\cdot \mu _ { k } ( k = 1 , 2 , 3 )$ 的计算:

$$
\mu _ { k } ^ { ( j + 1 ) } = \frac { \sum _ { i = 1 } ^ { N } P _ { i k } \hat { b } _ { i k } } { \sum _ { i = 1 } ^ { N } P _ { i k } }
$$

（3）关于参数 $\hat { \alpha }$ 的计算：

$$
\widehat { \boldsymbol { \alpha } } ^ { ( j + 1 ) } = \left( \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { 3 } \boldsymbol { W _ { i } } ^ { T } \boldsymbol { R } ^ { - 1 } \boldsymbol { W _ { i } } \right) ^ { - 1 } \left( \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { 3 } P _ { i k } \boldsymbol { W _ { i } } ^ { T } \boldsymbol { R } ^ { - 1 } \big ( y _ { i } - \mathbf { 1 } \widehat { b } _ { i k } \big ) \right)
$$

（4）关于参数 $\sigma _ { b } ^ { 2 }$ 、 $\sigma _ { \varepsilon } ^ { 2 }$ 的计算：

$$
\sigma _ { b } ^ { 2 } ^ { ( j + 1 ) } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { 3 } ( P _ { i k } \big ( \widehat { b } _ { i k } - \mu _ { k } \big ) ^ { 2 } + \widehat { \sigma } _ { b } ^ { 2 } )
$$

$$
\sigma _ { \varepsilon } ^ { 2 ( j + 1 ) } = \frac { 1 } { M } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { 3 } ( P _ { i k } ( ( y _ { i } - W _ { i } \hat { \alpha } - \mathbf { 1 } \hat { b } _ { i k } ) ^ { T } R ^ { - 1 } \big ( y _ { i } - W _ { i } \hat { \alpha } - \mathbf { 1 } \hat { b } _ { i k } \big ) + t r ( R ^ { - 1 } \mathbf { 1 } \hat { \sigma } _ { b } ^ { 2 } \mathbf { 1 } ^ { T } ) )
$$

（5）关于参数 $R$ 的计算：

$$
R ^ { ( j + 1 ) } = \frac { 1 } { N \sigma _ { \mathscr { E } } ^ { 2 } } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { 3 } P _ { i k } ( \left( y _ { i } - W _ { i } \widehat { \alpha } - \mathbf { 1 } \widehat { b } _ { i k } \right) ( y _ { i } - W _ { i } \widehat { \alpha } - \mathbf { 1 } \widehat { b } _ { i k } ) ^ { T } + \mathbf { 1 } \widehat { \sigma } _ { b } ^ { 2 } \mathbf { 1 } ^ { T } )
$$

# 3.2.3 似然比检验验证多项Logistic 模型中与分组概率有关的协变量

原假设（ $( \mathsf { H O } )$ ）：简化模型与完整模型在拟合观测数据上无显著差异备择假设（H1）：完整模型比简化模型对观测数据的拟合优度显著更好观测数据的对数似然函数

$$
l _ { f u l l } = \sum _ { i = 1 } ^ { N } l o g \left( \sum _ { k = 1 } ^ { 3 } \pi ( U _ { i } \gamma _ { k } ) \phi ( y _ { i } , W _ { i } \alpha + \mathbf { 1 } \mu _ { k } , \boldsymbol { \Sigma } ) \right)
$$

$$
l _ { r e d u c e d } = \sum _ { i = 1 } ^ { N } l o g \left( \sum _ { k = 1 } ^ { 3 } \pi \big ( U ^ { * } { } _ { i } \gamma ^ { * } { } _ { k } \big ) \phi \big ( y _ { i } , W _ { i } \alpha ^ { * } + \mathbf { 1 } { \mu ^ { * } } _ { k } , { \Sigma ^ { * } } \big ) \right)
$$

检验统计量：

$$
\mathrm { L R } = 2 ( l _ { f u l l } - l _ { r e d u c e d } ) { \dot { \sim } } \chi ^ { 2 } ( d f )
$$

其中 $\begin{array} { r } { \pi ( x ) = \frac { e ^ { x } } { \sum _ { k = 1 } ^ { 3 } e ^ { U _ { i } \gamma _ { k } } } ; } \end{array}$ 表示由多项逻辑回归求得的隶属各子组的先验概率。 $\phi ( y _ { i } , W _ { i } \alpha +$ $\mathbf { 1 } \mu _ { k } , \Sigma )$ 表示期望为 $W _ { i } \alpha + \mathbf { 1 } \mu _ { k }$ ，方差为 $\boldsymbol { \Sigma } = \mathbf { 1 } \sigma _ { b } ^ { 2 } \mathbf { 1 } ^ { T } + \sigma _ { \varepsilon } ^ { 2 } R _ { t \times t }$ ， $y _ { i }$ 的多元正态分布的密度函数。$l _ { f u l l }$ 完整模型的对数似然函数， $l _ { r e d u c e d }$ 表示简化模型的对数似然函数，也就是不包含全部协变量的模型。其中， $U _ { i }$ 表示所有观测到的协变量， $\gamma _ { k } , \alpha , \mu _ { k }$ 和 $\Sigma$ 是基于 $U _ { i }$ 中包含的协变量通过 EM 算法估计的参数值； $U ^ { * } { } _ { i }$ 就表示剔除一部分协变量之后剩下的认为与分组有关的协变量， $\gamma _ { \textbf { \textit { k } } } ^ { * }$ 、 $\alpha ^ { * }$ 、 $\boldsymbol { \mu ^ { * } } _ { k }$ 和 $\Sigma ^ { * }$ 是基于 $U ^ { * } { } _ { i }$ 中包含的协变量通过 EM 算法估计的参数值。 $d f$ 表示完整模型与简化模型的协变量数量之差
        loss = L_rec + lam1*L_deriv + lam2*L_curv + lam3*L_cons + lam4*L_T
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

