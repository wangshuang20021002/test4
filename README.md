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

1) 模型设定
1.1 潜在子组与随机效应

子组：对每个个体 i，引入潜在子组记号 δ_i ∈ {1, 2, 3}。

随机效应：给定子组 k，随机效应 b_i ~ Normal(mean = μ_k, var = σ_b^2)。三个子组均值不同（μ_1, μ_2, μ_3），但随机效应方差共享 σ_b^2。

1.2 子组概率（多项逻辑回归）

用与分组相关的协变量 U_i 通过多项 Logistic 模型定义隶属概率：
P(δ_i = k | U_i) = exp(U_i · γ_k) / Σ_{m=1..3} exp(U_i · γ_m)，k = 1,2,3。

上式允许不同个体因 U_i 的差异而有不同的子组概率。

补充视角：也可把 b 的边际分布写为有限混合：g(b) = Σ_{k=1..3} p_k · g_k(b)，其中 g_k 为以 μ_k、σ_b^2 的正态密度，p_k 为权重；若采用多项 Logistic，上述 p_k 由 U_i 驱动。

1.3 观测模型（纵向/重复测量）

设第 i 个体有 t 次观测，固定效应设计矩阵 W_i ∈ R^{t×p}，固定效应参数 α ∈ R^p，1 表示 t 维全 1 列向量。

观测向量 y_i 满足
y_i = W_i α + 1 · b_i + ε_i， 其中 ε_i ~ Normal(mean = 0, cov = σ_ε^2 · R)。
这里 R 是 t×t 的工作相关矩阵（平衡数据场景下 R 为对称，且 (R^{-1})^T = R^{-1}）。随机效应 b_i 与误差 ε_i 独立。

联合分布：在给定 δ_i、U_i、W_i 下，(y_i, b_i) 联合正态。
Var(y_i) = Σ = 1 σ_b^2 1^T + σ_ε^2 R；Cov(y_i, b_i) = B = 1 σ_b^2。
从而
E[b_i | y_i, δ_i = k] = μ_k + B^T Σ^{-1} (y_i − W_i α − 1 μ_k)，
Var[b_i | y_i, δ_i] = σ_b^2 − B^T Σ^{-1} B。

2) 参数估计与分组（EM 算法）

将 (b_i, δ_i) 视为缺失数据，整体参数记为
θ = (γ_1, γ_2, γ_3, μ_1, μ_2, μ_3, α, σ_ε^2, σ_b^2, R)。

2.1 完全数据对数似然（结构）

由三部分相加：
(a) 子组隶属的多项 Logistic 似然；
(b) 条件于 b_i 的 y_i 的多元正态密度；
(c) 条件于子组的 b_i 的正态密度。

2.2 E 步（计算“软分配”与 b 的后验）

计算后验子组概率（responsibility）：
P_i k = P(δ_i = k | y_i; θ^{(j)})
= [ π(U_i γ_k^{(j)}) · Normal(y_i ; mean = W_i α^{(j)} + 1 μ_k^{(j)}, cov = Σ^{(j)}) ]
/ Σ_{m=1..3} [ π(U_i γ_m^{(j)}) · Normal(y_i ; W_i α^{(j)} + 1 μ_m^{(j)}, Σ^{(j)}) ]。

计算每个子组下的 b 的后验均值与方差：
b_hat_{ik} = E[b_i | y_i, δ_i = k]； var_b_hat = Var[b_i | y_i, δ_i]（与上节公式一致）。

将 Q(θ, θ^{(j)}) 写为 I1 + I2 + I3 的和，I1 对应 Logistic 部分，I2 为 y|b 的二次型项，I3 为 b 的正态项。

2.3 M 步（更新参数）

子组隶属参数 γ_1, γ_2, γ_3：
最大化加权的多项 Logistic 对数似然：Σ_i [ P_i1 log π(U_i γ_1) + P_i2 log π(U_i γ_2) + P_i3 log π(U_i γ_3) ]。

子组均值 μ_k（k=1,2,3）：
μ_k ← (Σ_i P_i k · b_hat_{ik}) / (Σ_i P_i k)。

固定效应 α（广义最小二乘形式）：
记 M = N·t。更新式可写成
α ← (Σ_i Σ_k W_i^T R^{-1} W_i)^{-1} · (Σ_i Σ_k P_i k · W_i^T R^{-1} (y_i − 1·b_hat_{ik}))。

方差分量：
σ_b^2 ← (1/N) · Σ_i Σ_k P_i k [ (b_hat_{ik} − μ_k)^2 + var_b_hat ]。
σ_ε^2 ← (1/M) · Σ_i Σ_k P_i k [ (y_i − W_i α − 1·b_hat_{ik})^T R^{-1} (y_i − W_i α − 1·b_hat_{ik}) ]

(1/M) · tr( R^{-1} · 1 var_b_hat 1^T )。

工作相关矩阵 R：
R ← (1 / (N · σ_ε^2)) · Σ_i Σ_k P_i k [ (y_i − W_i α − 1·b_hat_{ik})(同上)^T + 1 var_b_hat 1^T ]。

2.4 个体分组与解释

软分配：用 P_i k 表示 i 属于子组 k 的后验概率。

硬分配：将个体 i 归入 argmax_k P_i k 的子组；并可汇报各子组规模、均值 μ_k 与方差分量等。

3) 模型检验与不确定性评估
3.1 协变量显著性（多项 Logistic 的似然比检验）

目的：验证与子组概率相关的协变量是否显著。

方法：比较“完整模型”（使用全部 U_i）与“简化模型”（只用 U_i* 的子集）的对数似然：
LR = 2 · ( loglik_full − loglik_reduced ) ≈ 逐步服从 χ^2，自由度为两模型协变量参数个数之差。

loglik_full 与 loglik_reduced 都基于 EM 估计的 (γ_k, α, μ_k, Σ) 计算，且 Σ = 1 σ_b^2 1^T + σ_ε^2 R。

3.2 标准误差与观测 Fisher 信息（Louis 方法）

在 EM 框架下，可用 Louis (1982) 的分解得到观测 Fisher 信息，从而给出参数估计的标准误差：
I_obs(θ) = E[−∂^2 log L_c | y] − E[s_c s_c^T | y] + E[s_c | y] E[s_c | y]^T，
其中 L_c 为完全数据似然、s_c 为完全数据得分。

该信息矩阵在本模型中呈分块结构（与混合 Logistic、均值 μ_k、方差分量与 R 等参数对应的块）。

4) 适用情景与假设小结

适用：有重复测量/纵向数据，存在“看不见的群体差异”（子组）导致个体间异质性的场景。

核心假设：

随机效应在各子组为正态，均值依子组而变、方差相同；

子组隶属概率由多项 Logistic 与 U_i 决定；

误差为多元正态，工作相关矩阵 R 正定且对称；

b_i 与 ε_i 相互独立；

数据平衡（每个体观测次数相同）。

5) 符号对照（便于实现）

i = 1..N（个体），t（每个体观测次数），p（固定效应个数）。

δ_i ∈ {1,2,3}：潜在子组；U_i：用于分组概率的协变量；γ_k：子组 k 的 Logistic 系数。

b_i：随机效应；μ_k：子组 k 的随机效应均值；σ_b^2：随机效应方差。

y_i ∈ R^t：观测向量；W_i ∈ R^{t×p}：设计矩阵；α ∈ R^p：固定效应参数。

ε_i ~ Normal(0, σ_ε^2 R)：误差；R：t×t 工作相关矩阵；Σ = 1 σ_b^2 1^T + σ_ε^2 R。

P_i k：后验隶属概率；b_hat_{ik}、var_b_hat：E 步得到的后验均值与方差。
