# Markets in a Spin / RARSI (Regime-Aware Regularised Spin Interaction) v4.1
 
A regime-aware cross-sectional return model inspired by statistical mechanics. RARSI combines adaptive PCA, three-state spin encoding, a Blume–Capel interaction model, a sticky Hidden Markov Model, online parameter learning, and risk-managed portfolio construction.
 
## 1. Model summary
 
Let $U=\left \lbrace 1,\dots,N\right \rbrace$ denote a universe of tradable assets and let $P_t^{(i)}>0$ be the adjusted close of asset $i$ at time $t$. The simple return is
 
$$
r_t^{(i)}=\frac{P_t^{(i)}}{P_{t-1}^{(i)}}-1,
\qquad
\mathbf{r}_t = \bigl(r_t^{(1)},\dots,r_t^{(N)}\bigr)^\top \in \mathbb{R}^N.
$$
 
The pipeline is:
 
1. compress $\mathbf{r}_t$ using adaptive PCA,
2. map latent scores into spins $s_j\in\left \lbrace -1,0,+1\right \rbrace$,
3. model cross-sectional dependence with a Blume–Capel Hamiltonian,
4. infer latent regimes with a log-space HMM,
5. learn parameters online via weighted pseudo-likelihood,
6. convert forecasts into portfolio weights with explicit risk controls.
## 2. Adaptive PCA
 
At walk-forward step $t$, define the rolling window
 
$$
\mathcal{W}_t=\left\lbrace \mathbf{r}_{t-W+1},\mathbf{r}_{t-W+2},\dots,\mathbf{r}_t \right\rbrace,
$$

 
and form the centered matrix
 
$$
\widetilde{\mathbf{R}}_t=\mathbf{R}_t-\mathbf{1}_W\boldsymbol{\mu}_t^\top,
\qquad
\boldsymbol{\mu}_t=\frac{1}{W}\mathbf{R}_t^\top \mathbf{1}_W.
$$
 
Using the singular value decomposition $\widetilde{\mathbf{R}}_t=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$, the loading matrix is
 
$$
\mathbf{P}_t=\mathbf{V}_{:,1:n}\in\mathbb{R}^{N\times n},
$$
 
and the latent score vector for a return observation $\mathbf{r}$ is
 
$$
\mathbf{z}=\mathbf{P}_t^\top(\mathbf{r}-\boldsymbol{\mu}_t)\in\mathbb{R}^n.
$$
 
## 3. Three-state spin encoding
 
Each principal component score $z_j$ is discretised into a spin variable $s_j\in\left \lbrace -1,0,+1 \right \rbrace$. A z-score thresholding rule is used. Let $\Phi$ be the standard normal CDF and let
 
$$
z^\star=\Phi^{-1}(1-q), \qquad q\in\left(0,\tfrac{1}{3}\right].
$$
 
With training standard deviation $\sigma_j=\text{std}(\{z_{t,j}\})$, the thresholds are
 
$$
\tau_j^{\mathrm{lo}}=-z^\star\sigma_j,
\qquad
\tau_j^{\mathrm{hi}}=+z^\star\sigma_j,
$$
 
and the encoding rule is
 
$$
s_j=
\begin{cases}
-1, & z_j<\tau_j^{\mathrm{lo}},\\
0, & \tau_j^{\mathrm{lo}}\le z_j\le \tau_j^{\mathrm{hi}},\\
+1, & z_j>\tau_j^{\mathrm{hi}}.
\end{cases}
$$
 
This preserves the interpretation of weak movements as neutral rather than forcing them into a directional class.
 
## 4. Blume–Capel interaction model
 
For a spin configuration $\mathbf{x}\in\left \lbrace -1,0,+1 \right \rbrace ^n$, the Blume–Capel Hamiltonian is
 
$$
H(\mathbf{x};\theta) = -\sum_{i=1}^n h_i x_i -\sum_{i=1}^n d_i x_i^2 -\sum_{1\le i \lt j\le n} J_{ij}x_i x_j,
$$

 
where $\theta=(\mathbf{h},\mathbf{d},\mathbf{J})$, $\mathbf{J}=\mathbf{J}^\top$, and $J_{ii}=0$. The associated distribution is
 
$$
p(\mathbf{x};\theta)=\frac{1}{Z(\theta)}\exp\\bigl(-H(\mathbf{x};\theta)\bigr),
\qquad
Z(\theta)=\sum_{\mathbf{x}\in\left \lbrace -1,0,+1 \right \rbrace ^n}\exp\\bigl(-H(\mathbf{x};\theta)\bigr).
$$
 
## 4.1 Mean-field approximation

Because exact inference is exponential in $n$, RARSI replaces the true posterior over spin configurations by a factorised variational family

$$q(\mathbf{x})=\prod_{i=1}^n q_i(x_i), \qquad x_i\in\left \lbrace -1,0,+1 \right \rbrace.$$

The mean-field equations are obtained by minimizing the Kullback–Leibler divergence from the variational distribution to the target model:

$$q^\star=\arg\min_{q\in\mathcal{Q}} D_{\mathrm{KL}}\\left(q(\mathbf{x})\\|p(\mathbf{x})\right),$$

where

$$p(\mathbf{x})=\frac{1}{Z}\exp\bigl(-H(\mathbf{x};\theta)\bigr).$$

Expanding the divergence gives

$$D_{\mathrm{KL}}(q\|p) = \sum_{\mathbf{x}} q(\mathbf{x})\log q(\mathbf{x}) + \mathbb{E}_q\\left[H(\mathbf{x};\theta)\right] + \log Z.$$

Since $\log Z$ does not depend on $q$, the optimization is equivalent to minimizing the variational free energy

$$\mathcal{F}[q] = \mathbb{E}_q\\left[H(\mathbf{x};\theta)\right] + \sum_{\mathbf{x}} q(\mathbf{x})\log q(\mathbf{x}).$$

Under the factorisation assumption, the entropy term decomposes as

$$\sum_{\mathbf{x}} q(\mathbf{x})\log q(\mathbf{x}) = \sum_{i=1}^n \sum_{x_i} q_i(x_i)\log q_i(x_i),$$

and the energy term depends on each node only through its first and second moments:

$$m_i=\mathbb{E}_q[x_i], \qquad q_i^{(2)}=\mathbb{E}_q[x_i^2].$$

For the Blume–Capel Hamiltonian

$$
H(\mathbf{x};\theta) = -\sum_{i=1}^n h_i x_i -\sum_{i=1}^n d_i x_i^2 -\sum_{1\le i \lt j\le n} J_{ij}x_i x_j,
$$

the variational objective becomes

$$\mathcal{F}[q] = -\sum_{i=1}^n h_i m_i -\sum_{i=1}^n d_i q_i^{(2)} -\sum_{i \lt j}J_{ij}m_i m_j + \sum_{i=1}^n \sum_{x_i} q_i(x_i)\log q_i(x_i).$$


Taking a functional derivative with respect to $q_i(x_i)$ and enforcing normalization with a Lagrange multiplier yields the stationary condition

$$\log q_i(x_i) = \text{const} + h_i x_i + d_i x_i^2 + x_i\sum_{j\ne i}J_{ij}m_j.$$

Defining the effective field

$$h_i^{\mathrm{eff}}=h_i+\sum_{j\ne i}J_{ij}m_j, \qquad m_j=\mathbb{E}_q[x_j],$$

we obtain the one-site marginals

$$q_i(+1)\propto e^{d_i+h_i^{\mathrm{eff}}}, \qquad q_i(0)\propto 1, \qquad q_i(-1)\propto e^{d_i-h_i^{\mathrm{eff}}}.$$

Hence

$$m_i=q_i(+1)-q_i(-1), \qquad q_i^{(2)}=\mathbb{E}_q[x_i^2]=q_i(+1)+q_i(-1)=1-q_i(0).$$

The self-consistency equations are solved by fixed-point iteration until

$$\left\|\mathbf{m}^{(\tau)}-\mathbf{m}^{(\tau-1)}\right\|_\infty<\varepsilon.$$
### 4.2 Pseudo-log-likelihood

For observed spins $\mathbf{x}$, the pseudo-log-likelihood is

$$
\mathrm{PLL}(\mathbf{x};\theta) = \sum_{i=1}^n \log p(x_i \mid \mathbf{x}_{-i};\theta).
$$

The useful gradients are

$$
\frac{\partial\\mathrm{PLL}}{\partial h_i}=x_i-m_i, \qquad \frac{\partial\\mathrm{PLL}}{\partial d_i}=x_i^2-q_i^{(2)}, \qquad \frac{\partial\\mathrm{PLL}}{\partial J_{ij}}=x_i x_j-m_i m_j \quad (i\ne j).
$$

 
## 5. Regime-switching Hidden Markov Model
 
Let $C_t\in\{1,\dots,K\}$ be the latent regime. The transition matrix $A\in\mathbb{R}^{K\times K}$ satisfies
 
$$
A_{ij}=P(C_t=j\mid C_{t-1}=i),
\qquad
\sum_{j=1}^K A_{ij}=1.
$$
 
Conditional on $C_t=k$, the spin configuration is emitted from a regime-specific Blume–Capel model:
 
$$
p(\mathbf{x}_t\mid C_t=k)=p_{\mathrm{BC}}(\mathbf{x}_t;\theta^{(k)}).
$$
 
### 5.1 Sticky transition prior
 
$$
A_{ij}=
\begin{cases}
\alpha, & i=j,\\
\dfrac{1-\alpha}{K-1}, & i\ne j,
\end{cases}
\qquad \alpha\in(0,1).
$$
 
### 5.2 Log-space filtering
 
Let $\alpha_{t,k}=P(C_t=k\mid \mathbf{x}_{1:t})$. To avoid underflow, the forward update is performed in log-space:
 
$$
\log \widetilde{\alpha}_{t,k} = \log\\left(\sum_{i=1}^K \exp(\log A_{ik}+\log \alpha_{t-1,i})\right) + \frac{\log e_{t,k}}{\tau},
$$
 
where $e_{t,k}=p(\mathbf{x}_t\mid C_t=k)$ and $\tau\ge 1$ is an emission temperature. Normalisation is then 

$$
\log \alpha_{t,k} = \log \widetilde{\alpha}_{t,k} - \mathrm{logsumexp}(\log \widetilde{\boldsymbol{\alpha}}_t).
$$


The $H$-step regime forecast is $\boldsymbol{\alpha}_{t+H}=A^H\boldsymbol{\alpha}_t$.

 
## 6. Online learning
 
The regime-weighted objective for regime $k$ is
 
$$
L_t^{(k)}(\theta^{(k)}) = w_{t,k}\\mathrm{PLL}(\mathbf{x}_t;\theta^{(k)}) - \Omega(\theta^{(k)}), \qquad w_{t,k}=\alpha_{t,k}.
$$
 
The regulariser is
 
$$
\Omega(\theta) = \frac{\lambda_2}{2}\lVert \mathbf{h}\rVert_2^2 + \frac{\lambda_2^{(d)}}{2}\lVert \mathbf{d}\rVert_2^2 + \frac{\lambda_2}{2}\lVert \mathbf{J}\rVert_F^2 + \lambda_1^{(J)}\lVert \mathbf{J}\rVert_{1,\mathrm{off\text{-}diag}}.
$$
 
AdaGrad-style preconditioning uses
 
$$
G_t=G_{t-1}+\mathrm{diag}(g_t\odot g_t), \qquad \theta_{t+1} = \theta_t - \eta\(G_t^{1/2}+\varepsilon I)^{-1}g_t.
$$
 
The transition matrix is updated by exponentiated gradient on posterior transition counts
 
$$
\xi_{ij} = \frac{\alpha_{t-1,i}A_{ij}e_{t,j}}{\sum_{k=1}^K \alpha_{t-1,i}A_{ik}e_{t,k}}.
$$

 
## 7. Signal generation
 
The regime-specific expected spin at horizon $H$ is

$$
\bar{\mathbf{s}}_{t+H} = \sum_{k=1}^K \alpha_{t+H,k}\\mathbf{m}^{(k)}.
$$

The Blume-Capel signal in asset space is then

 
$$
\mathbf{s}_t^{\mathrm{BC}}=\mathbf{P}_t\\bar{\mathbf{s}}_{t+H}.
$$
 
A momentum signal is computed on multiple lookback windows $W=(w_1,w_2,w_3)$:
 
$$
m_t^{(i)}(w)=\frac{1}{w}\sum_{\tau=0}^{w-1} r_{t-\tau}^{(i)},
$$
 
standardised cross-sectionally and averaged across windows. The final blend uses rolling Information Coefficients, i.e. Spearman rank correlations $\rho_S(\mathbf{s},\mathbf{r})$.
 
## 8. Portfolio construction
 
Given a signal $\mathbf{s}$, raw weights may be formed by
 
$$
w_i=s_i\\mathbf{1}\\left[\lvert s_i\rvert>\theta_{\mathrm{entry}}\right],
$$
 
or by long-only, equal-direction, or volatility-scaled variants such as
 
$$
w_i=\frac{s_i}{\sigma_i+10^{-8}}\\mathbf{1}\\left[\lvert s_i\rvert>\theta_{\mathrm{entry}}\right].
$$
 
Risk management layers then apply:
 
$$
\mathbf{w} \leftarrow \frac{\mathbf{w}}{\lVert \mathbf{w}\rVert_1},
\qquad
w_i \leftarrow \text{clip}(w_i,-w_{\max},w_{\max}),
$$
 
followed by re-normalisation. Volatility targeting uses
 
$$
L_t=\text{clip}\\left(\frac{\sigma_{\mathrm{target}}}{\sigma_{\mathrm{ann}}+10^{-6}}\,0.1\,L_{\max}\right).
$$
 
Turnover and transaction costs are
 
$$
\mathrm{TO}_t=\lVert \mathbf{w}_t-\mathbf{w}_{t-1}\rVert_1,
\qquad
\mathrm{Cost}_t=c\\mathrm{TO}_t.
$$
 
## 9. Walk-forward protocol
 
At each out-of-sample step $t\in\left \lbrace W,\dots,T-H\right \rbrace$:
 
1. fit PCA on $\mathcal{W}_t$,
2. encode latent scores into spins,
3. fit the regime-specific Blume–Capel models,
4. infer $\boldsymbol{\alpha}_t$ with the HMM filter,
5. update parameters online,
6. generate forecasts,
7. convert forecasts into portfolio weights,
8. evaluate on future returns only.
This ensures true walk-forward validation: no peeking, no future leakage, and no in-sample evaluation.
 
## 10. Performance metrics
 
Annualised return and volatility are
 
$$
\mu_{\mathrm{ann}}=(1+\bar{r})^{252}-1, \qquad \sigma_{\mathrm{ann}}=\mathrm{std}(\{r_t\})\sqrt{252}.
$$
 
The Sharpe ratio is
 
$$
\mathrm{Sharpe} = \frac{\bar{r}-r_f}{\mathrm{std}(\{r_t-r_f\})}\sqrt{252}, \qquad r_f=\frac{0.065}{252}.
$$
 
Maximum drawdown and Calmar ratio are
 
$$
\mathrm{MDD}=\min_t \frac{R_t-\max_{\tau\le t}R_\tau}{\max_{\tau\le t}R_\tau}, \qquad \mathrm{Calmar}=\frac{\mu_{\mathrm{ann}}}{|\mathrm{MDD}|}.
$$
 
Directional accuracy and IC-IR are defined analogously from realised returns and predicted signals.

 
## 11. Computational complexity
 
| Operation | Complexity |
|---|---:|
| PCA / SVD | $O(\min(W^2N,WN^2))$ |
| Spin encoding | $O(Nn)$ |
| Mean-field iteration (per regime) | $O(I_{\max}n^2)$ |
| PLL gradient (per regime) | $O(n^2)$ |
| HMM forward step | $O(K^2)$ |
| OMD update (per regime) | $O(n^2)$ |
| Transition EG update | $O(K^2)$ |
 
## 12. Numerical stability checklist
 
- log-sum-exp normalisation for all probabilities,
- emission clipping before exponentiation,
- gradient clipping for online updates,
- Frobenius projection for $\mathbf{J}$,
- belief flooring in the HMM,
- mean-zero centering of $\mathbf{d}$,
- SVD-based PCA with stable truncation.
## 13. References
 
- Blume, M. (1966).
- Capel, H. W. (1966).
- Hyvärinen, A. (2006).
- Cappé, O. and Moulines, E. (2009).
- Duchi, J., Hazan, E., and Singer, Y. (2011).
- Kivinen, J. and Warmuth, M. K. (1997).
- Rabiner, L. R. (1989).
- Jolliffe, I. T. (2002).
 
