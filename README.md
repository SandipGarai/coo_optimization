# **Canine Olfactory Optimizer (COO): A Novel Bio-Inspired Metaheuristic Algorithm for optimization**

## 1. **Biological Inspiration**

Dogs use their sense of smell to:

- **Track scents** using concentration gradients (exploitation)
- **Cast about** in zigzag patterns when losing a trail (exploration)
- **Cooperate in packs** to cover more ground and share information
- **Return to strong scent sources** (memory-based search)

## 2. **Mathematical Formulation**

### **2.1 Problem Definition**

Maximize: $f(\mathbf{x})$ where $\mathbf{x} \in \mathbb{R}^d$

Subject to: $\mathbf{x}_i \in [L_i, U_i]$ for $i = 1, \ldots, d$

Where:

- $f(\mathbf{x})$ = "scent strength" (objective function)
- $\mathbf{x}$ = position in search space
- $d$ = dimensionality

### **2.2 Population Structure**

**Multi-Pack System:**

$$P = \{P_1, P_2, \ldots, P_k\}$$

Where pack $P_j$ contains $n_j$ individuals (dogs):
$$P_j = \{\mathbf{x}_j^1, \mathbf{x}_j^2, \ldots, \mathbf{x}_j^{n_j}\}$$

**Practical Relevance:** Multiple packs explore different regions simultaneously, preventing premature convergence (like dogs searching different areas of a field).

### **2.3 Core Movement Equations**

For each dog $i$ in pack $j$ at iteration $t$:

#### **Velocity Update (Momentum + Attraction):**

$$\mathbf{v}_i^{t+1} = \omega \mathbf{v}_i^t + \alpha (\mathbf{x}_{local}^* - \mathbf{x}_i^t) + \beta(t) (\mathbf{x}_{global}^* - \mathbf{x}_i^t)$$

Where:

- $\omega = 0.6$ = momentum weight (inertia from previous movement)
- $\alpha = 0.3$ = local attraction coefficient (pack-local best)
- $\beta(t) = 0.4 + 0.6 \cdot \frac{t}{T_{max}}$ = adaptive cooperation weight
- $\mathbf{x}_{local}^*$ = best position in current pack
- $\mathbf{x}_{global}^*$ = global best position across all packs
- $T_{max}$ = maximum iterations

**Practical Relevance:** 

- **Momentum** ($\omega \mathbf{v}_i^t$): Dogs maintain movement direction (physical inertia)
- **Local attraction** ($\alpha$): Dogs follow strong scents found by packmates nearby
- **Global attraction** ($\beta(t)$): Increasing cooperation over time as confidence in best location grows

#### **Position Update with Sniffing Noise:**

$$\mathbf{x}_i^{t+1} = \mathbf{x}_i^t + \mathbf{v}_i^{t+1} + \sigma_1(t) \boldsymbol{\epsilon}_1$$

Where:

- $\sigma_1(t) = \sigma_1^{init} \cdot e^{-0.05t}$ = exploration radius (exponential decay)
- $\boldsymbol{\epsilon}_1 \sim \mathcal{N}(0, \mathbf{I})$ = Gaussian random noise
- $\sigma_1^{init} = 0.35$ = initial exploration strength

**Practical Relevance:** The $\sigma_1(t) \boldsymbol{\epsilon}_1$ term models **sniffing behavior** - random local exploration that decreases as the search progresses (exploitation vs exploration balance).

#### **Zigzag Reacquisition Movement:**

With probability $p_{zigzag} = 0.18$:

$$\mathbf{x}_i^{t+1} = \mathbf{x}_i^{t+1} + \sigma_2(t) \boldsymbol{\epsilon}_2 \cdot \sin\left(2\pi \frac{t}{T_{max}}\right)$$

Where:

- $\sigma_2(t) = \sigma_2^{init} \cdot e^{-0.07t}$ 
- $\sigma_2^{init} = 0.12$
- $\boldsymbol{\epsilon}_2 \sim \mathcal{N}(0, \mathbf{I})$

**Practical Relevance:** When dogs lose a scent trail, they perform **casting behavior** - zigzagging to reacquire the scent. The sinusoidal term creates periodic oscillations mimicking this biological pattern.

### **2.4 Gradient-Based Refinement**

For top $\rho \cdot N$ individuals (where $\rho = 0.10$):

$$\nabla f(\mathbf{x}) \approx \left[\frac{f(\mathbf{x} + \epsilon \mathbf{e}_j) - f(\mathbf{x} - \epsilon \mathbf{e}_j)}{2\epsilon}\right]_{j=1}^d$$

**Gradient ascent step:**

$$\mathbf{x}_{refined} = \mathbf{x} + \eta \frac{\nabla f(\mathbf{x})}{\|\nabla f(\mathbf{x})\| + 10^{-12}}$$

Where:

- $\epsilon = 10^{-4}$ = finite difference step
- $\eta = 0.03$ = learning rate (trust region)
- Step is clipped to $[-0.07, 0.07]$

**Practical Relevance:** Elite dogs (those with strongest scent) perform **focused investigation** - systematic local search along the gradient direction (uphill toward stronger scent).

### **2.5 Elitist Exchange Mechanism**

Every $k_{exchange} = 6$ iterations:

1. **Rank all pack-local bests:**

$$
   B = \{ (\mathbf{x}_{1}^{*}, f_{1}^{*}), (\mathbf{x}_{2}^{*}, f_{2}^{*}), \ldots, (\mathbf{x}_{k}^{*}, f_{k}^{*}) \}.

$$

2. **Select top-3 elites:**

$$
   E = \{\mathbf{x}_{(1)}^{*},\; \mathbf{x}_{(2)}^{*},\; \mathbf{x}_{(3)}^{*}\}.
$$

3. **Inject perturbed elites into each pack:**

$$
   \mathbf{x}_{worst} \leftarrow \mathbf{x}_{elite} + 0.01 \cdot \boldsymbol{\epsilon}
$$

**Practical Relevance:** This models **information sharing between packs** through vocalizations or visual cues. Weak-performing dogs are replaced with perturbed versions of successful strategies from other packs.

### **2.6 Adaptive Pack Sizing**

If no improvement for $> 10$ iterations:
$$n_j^{t+1} = \max\{n_{min}, n_j^t - 1\}$$

**Practical Relevance:** When the search stagnates, reduce pack sizes (fewer agents exploring the same region) to conserve computational resources and focus exploitation.

### **2.7 Surrogate-Assisted Evaluation**

Train ensemble model $\hat{f}$ every $k_{retrain} = 5$ iterations:
$$\hat{f}(\mathbf{x}) = \frac{1}{M} \sum_{m=1}^M f_m(\mathbf{x})$$

Where $f_m$ are base models (Random Forest, Gradient Boosting).

**Selective evaluation strategy:**

- Evaluate all positions with $\hat{f}(\mathbf{x})$
- Perform exact $f(\mathbf{x})$ only on top 50%

**Practical Relevance:** Dogs don't exhaustively sniff everywhere - they use **quick assessments** (surrogate) to identify promising areas, then investigate thoroughly (exact evaluation) only the best candidates.

## 3. **Algorithm Complexity**

**Time per iteration:**

- Velocity/position updates: $O(k \cdot n \cdot d)$
- Fitness evaluations: $O(k \cdot n \cdot C_f)$ where $C_f$ = cost of $f(\mathbf{x})$
- Gradient refinement: $O(\rho \cdot k \cdot n \cdot d \cdot C_f)$
- Surrogate training: $O(N_{train}^2 \cdot d)$ (amortized)

**Total:** $O(T_{max} \cdot k \cdot n \cdot (d + C_f))$

## 4. **Key Algorithmic Properties**

| **Property** | **Mechanism** | **Mathematical Guarantee** |
|--------------|---------------|----------------------------|
| **Global Convergence** | Multi-pack exploration + gradient refinement | Ergodic (visits entire space with non-zero probability) |
| **Exploitation** | Increasing $\beta(t)$, decreasing $\sigma_1(t)$, $\sigma_2(t)$ | Controlled by cooling schedules |
| **Diversity** | Multiple packs, zigzag movements | Prevents premature convergence |
| **Computational Efficiency** | Caching, surrogate models | Reduces redundant $f(\mathbf{x})$ evaluations |

## 5. **Biological-Mathematical Correspondence**

| **Dog Behavior** | **Mathematical Model** | **Parameter** |
|------------------|------------------------|---------------|
| Following scent gradient | Velocity toward $\mathbf{x}_{local}^*$, $\mathbf{x}_{global}^*$ | $\alpha$, $\beta(t)$ |
| Sniffing around | Gaussian noise $\sigma_1(t) \boldsymbol{\epsilon}_1$ | $\sigma_1 = 0.35$ |
| Casting (zigzag when lost) | Periodic perturbation $\sigma_2(t) \sin(2\pi t/T)$ | $p_{zigzag} = 0.18$ |
| Pack cooperation | Multi-agent system with information sharing | $k$ packs |
| Memory of strong scents | Global best tracking $\mathbf{x}_{global}^*$ | Persistent memory |
| Territorial hunting | Multiple non-overlapping packs | $k = 2$ packs |

This is a well-designed hybrid algorithm combining:

1. **Swarm intelligence** (PSO-like movement)
2. **Evolutionary strategies** (elitist selection)
3. **Local search** (gradient refinement)
4. **Surrogate modeling** (computational efficiency)

# **About the website** 

## **Section 1: Biological Inspiration**

- **Visual simulation** of dogs finding food using their sense of smell
- Shows scent gradient visualization (green glow around food)
- Dogs exhibit realistic behaviors: following gradients, zigzagging when lost, cooperation
- Real-time stats: number of dogs searching, average distance to food, dogs that found food

## **Section 2: Algorithm Visualization**

- **Two tabs** for organizing parameters:

  - **Basic Parameters**: Pack configuration, momentum, attraction weights, exploration
  - **Advanced Parameters**: Zigzag behavior, elitist exchange, gradient refinement toggles
  
- **Interactive controls** for all hyperparameters with real-time value displays
- **Visual simulation** showing multiple packs (different colors) converging to target
- **Four action buttons**:

  - Run Optimization
  - Reset
  - Random Hyperparameters (test random configurations)
  - Find Best Hyperparameters (auto-tune)
  
- **Real-time statistics**: iteration count, best fitness, evaluations, convergence rate

## **Section 3: Rosenbrock Function Optimization**

- **3D colorful visualization** of the Rosenbrock function surface
- Color-coded height map (red = high values, blue/green = low values)
- **Interactive 3D view controls**: rotate angle and elevation to see from different perspectives
- Green marker shows global minimum at (1, 1)
- Dogs shown as colored dots on the surface
- Golden ring highlights current best position
- **Statistics tracked**: best position found, function value, distance to optimum, success rate
- Auto-tune button optimized specifically for Rosenbrock function characteristics
