# 7-scientific-ML-goodies-from-shannon-to-einstein

A high-performance Machine Learning pipeline built on top of the Airline Passenger Satisfaction dataset. Instead of using traditional, linear transformations, this project introduces a breakthrough approach by mapping tabular customer behavior to 7 fundamental concepts of theoretical physics, information theory, and network topology. The final model utilizes a robust 5-Fold Cross-Validation Triple Ensemble consisting of CatBoost, XGBoost, and LightGBM, achieving a highly stable and cross-validated Out-of-Fold (OOF) ROC-AUC score of 0.93792.

---

## Core Scientific Modules (The 7 Goodies)

This repository demonstrates how complex equations from classical and modern science can be fully vectorized via NumPy to extract non-linear patterns that traditional decision trees struggle to find in raw form.

### 1. Information Theory (Shannon Entropy)
Measures the emotional chaos and contradiction in passenger feedback. By calculating the probability distribution of text ratings across various service metrics, the pipeline detects high-entropy profiles where passengers experience volatile shifts in emotion.
The entropy \(H(X)\) for a distribution of ratings is computed as:

$$H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)$$

### 2. Airline Thermodynamics (Gibbs Free Energy)
Models passenger loyalty as a closed thermodynamic system. Enthalpy (\(H\)) represents the base comfort level of core services, Entropy (\(S\)) captures the chaotic decay introduced by delays, and Temperature (\(T\)) modulates sensitivity based on the travel context (Personal vs. Business).
The stability of the passenger system is governed by the Gibbs equation:

$$\Delta G = \Delta H - T \Delta S$$

### 3. Graph Theory & Network Topology (Digital Centrality)
Constructs a localized spatial network where individual service ratings act as nodes. It calculates the Closeness Centrality of the core digital infrastructure (Inflight Wi-Fi) to map how digital satisfaction propagates through physical comfort nodes.
The centrality index is defined as:

$$C(u) = \frac{1}{\sum_{v \neq u} d(u,v) + 1}$$

### 4. Evolutionary Genetics (Crossover & Mutation)
Simulates a single-generation genetic algorithm operating directly at the CPU bitwise level. It applies a crossover mask to combine the binary chromosomes of top-performing features, introducing a stochastic mutation probability to discover completely novel synthetic attributes.
The evolution step uses bitwise operations:

$$\text{Chromosome}_{\text{new}} = (\text{Gene}_A \wedge \text{Mask}_1) \vee (\text{Gene}_B \wedge \text{Mask}_2) \oplus \text{Mutation}SS

### 5. Chaos Theory & Nonlinear Dynamics (Lorenz Strange Attractors)
Maps passenger states into a three-dimensional phase space regulated by the Lorenz system of differential equations. It vectorizes the velocities of behavioral changes and computes the exact Euclidean distance to the turbulent center of attraction, predicting sudden phase transitions into dissatisfaction.
The system velocities are defined by:

$$\frac{dx}{dt} = \sigma(y - x), \quad \frac{dy}{dt} = x(\rho - z) - y, \quad \frac{dz}{dt} = xy - \beta z$$

### 6. String Theory & Supersymmetry (SUSY Resonant Index)
Treats individual services as specific vibration modes of an open quantum string. By building a multidimensional tension tensor out of variance anomalies, it computes a Supersymmetric Resonant Index that flags harmonic alignments or destructive asymmetric feedback loops in the cabin experience.
The resonance is computed using hyperbolic tangent scaling:

$$I_{\text{SUSY}} = \tanh\left(\frac{(d_{\text{wifi}} \cdot d_{\text{food}}) + (d_{\text{seat}} \cdot d_{\text{entertainment}}) - (d_{\text{wifi}} \cdot d_{\text{seat}})}{10}\right)$$

### 7. General Relativity (Einsteinian Time Dilation)
Models heavy flight delays as a localized gravitational mass creating a psychological black hole. By implementing a Schwarzschild metric approximation, it computes a relativistic Time Dilation Factor to quantify how waiting time subjectively stretches for a delayed passenger.
The psychological time dilation factor is calculated as:

$$t_{\text{dilation}} = \frac{1}{\sqrt{1 + \frac{M_{\text{delay}}}{\alpha}}}$$

---

## Performance Matrix (5-Fold CV Out-of-Fold Results)

The features show immense emergent synergy when trained together, significantly outperforming individual baseline features.

* Baseline Core Metrics Control: 0.80560 ROC-AUC
* Shannon Opinion Entropy Isolation: 0.85758 ROC-AUC
* String SUSY Resonance Isolation: 0.85652 ROC-AUC
* Graph Topology Centrality Isolation: 0.87755 ROC-AUC
* Einsteinian Time Dilation Isolation: 0.90827 ROC-AUC
* Thermodynamics Gibbs Energy Isolation: 0.92682 ROC-AUC
* Lorenz Chaos Attractor Isolation: 0.92996 ROC-AUC
* **Final Triple Ensemble OOF Synergy: 0.93792 ROC-AUC**

---

## Installation & Basic Usage

Ensure you have XGBoost, CatBoost, and LightGBM installed in your Python environment:

```bash
pip install xgboost catboost lightgbm pandas numpy scikit-learn
```

Run the pipeline by executing the main script with `train.csv` and `test.csv` present in your active working directory:

```python
# Load the dataset and run the pipeline
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Apply all 7 scientific features vectorized via NumPy
train_df = apply_all_scientific_goodies(train_df)
test_df = apply_all_scientific_goodies(test_df)
```

