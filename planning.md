# Performance & Optimization Planning

## 1. Where the Time Goes (Bottleneck Analysis)

### The Core Loop (the real culprit)

In `simulate_three_photon_rabi_dynamics` (ERF mode, `n_shots=50`, `N_atoms=50`):

```
for i, t_pulse in enumerate(tlist):          # 50 shots
    for j in range(N_atoms_i):               # 50 atoms
        qt.mesolve(H_sys, rho0, tlist_sim, c_ops, e_ops)   # ← called 2500 times
```

Each `mesolve` call on a **5×5** density matrix carries:
- QuTiP object construction overhead (Python-level `Qobj` allocation)
- Adaptive ODE solver spin-up (VODE/Adams method from scipy under the hood)
- Python callback overhead per time step for the time-dependent envelope coefficients (`c0, c1, c2` are closures evaluated at every internal step)

Total: ~2500 independent ODE solves, each paying full QuTiP overhead. This is the dominant cost.

### Secondary Costs

| Source | Cost | Notes |
|--------|------|-------|
| QuTiP object construction per atom | Medium | `H_det`, `H_sys` rebuilt every atom |
| Envelope closure calls inside mesolve | Medium | Python-level callback at each ODE step |
| `get_calculated_parameters` per shot | Low | Vectorized numpy, negligible |
| `sample_atomic_ensemble` per shot | Low | Negligible |

---

## 2. Speedup Strategies (Ranked by Impact)

### Strategy A: Replace QuTiP with Direct Numpy/SciPy ODE Integration ⭐⭐⭐⭐⭐

**Why**: QuTiP is designed for arbitrary-dimensional, symbolically structured Hamiltonians. For a fixed 5-level system, it is massive overkill. The density matrix is just a 5×5 complex matrix (25 complex numbers). You can write the Lindblad RHS directly as a matrix–vector product using the superoperator (vectorized) form, then hand it to `scipy.integrate.solve_ivp` or a hand-rolled RK4.

The Lindblad equation in vectorized form:
```
drho/dt = L @ rho_vec
```
where `L` is a 25×25 complex matrix (the Liouvillian superoperator) built once per atom from `H` and `c_ops`. This removes all QuTiP overhead.

**Expected speedup**: 10–50× per mesolve call.

**Implementation sketch**:
```python
import numpy as np
from scipy.integrate import solve_ivp

def build_liouvillian(H, c_ops):
    """Build 25x25 Liouvillian for a 5-level system."""
    d = H.shape[0]
    I = np.eye(d)
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for c in c_ops:
        cdag_c = c.conj().T @ c
        L += np.kron(c.conj(), c) - 0.5 * np.kron(I, cdag_c) - 0.5 * np.kron(cdag_c.T, I)
    return L

def mesolve_numpy(H_const, H_td_pairs, rho0, tlist, c_ops):
    """
    H_td_pairs: list of (H_matrix, coeff_func) for time-dependent terms.
    Returns populations at each time step.
    """
    d = rho0.shape[0]
    L0 = build_liouvillian(H_const, c_ops)
    rho0_vec = rho0.flatten()

    def rhs(t, rho_vec):
        L = L0.copy()
        for H_op, coeff_fn in H_td_pairs:
            c = coeff_fn(t)
            L += c * build_liouvillian(H_op, [])
        return L @ rho_vec

    sol = solve_ivp(rhs, [tlist[0], tlist[-1]], rho0_vec,
                    method='RK45', t_eval=tlist, rtol=1e-6, atol=1e-8)
    return sol.y  # shape (25, n_steps)
```

For the **SQUARE** (constant H) case, `L` is time-independent and the solution is `expm(L * t) @ rho0_vec` — computable analytically via matrix exponential, making each atom's simulation a single `scipy.linalg.expm` call, not an ODE integration.

---

### Strategy B: Vectorize Across Atoms ⭐⭐⭐⭐

**Why**: Each atom sees a different `(O0, O1, O2, d0, d01, d012)` tuple, but the **structure** of the ODE is identical. With numpy arrays, you can batch all atoms into a single vectorized ODE:

```python
# rho has shape (N_atoms, 25) — all atoms propagated simultaneously
def rhs_batched(t, rho_batch):
    # rho_batch shape: (N_atoms * 25,) — flattened for solve_ivp
    rho = rho_batch.reshape(N_atoms, 25)
    drho = np.einsum('aij,aj->ai', L_batch, rho)   # L_batch shape (N_atoms, 25, 25)
    return drho.flatten()
```

For the SQUARE case this is exact: build one Liouvillian per atom (vectorized construction), then batch-`expm` or use a single `solve_ivp` call.

**Expected speedup over A alone**: 2–5× additional (better cache use, single ODE solver spin-up per shot).

---

### Strategy C: Parallelize Across Shots with `joblib` ⭐⭐⭐

**Why**: Each shot (pulse duration) is entirely independent. This is embarrassingly parallel.

```python
from joblib import Parallel, delayed

def simulate_one_shot(t_pulse, ...):
    # all the per-shot logic
    return shot_pop

results = Parallel(n_jobs=-1)(
    delayed(simulate_one_shot)(t_pulse, ...)
    for t_pulse in tlist
)
```

`n_jobs=-1` uses all CPU cores. On an 8-core machine this gives ~8× wall-clock speedup for free, with no algorithmic changes.

**Caveat**: QuTiP is not fork-safe on all platforms (Windows especially). Combine with Strategy A first to avoid this issue.

---

### Strategy D: JAX JIT + Vectorization ⭐⭐⭐⭐ (for optimization use case)

**Why**: JAX compiles numerical code to XLA (runs on CPU or GPU), supports `jit`, `vmap` (vectorize over any axis), and `grad` (automatic differentiation). This is the best option if you want **gradient-based parameter optimization**.

```python
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

@jax.jit
def simulate_atom(O0, O1, O2, d0, d01, d012, t_max):
    H = build_H_jax(O0, O1, O2, d0, d01, d012)
    L = build_liouvillian_jax(H, c_ops)
    rho_final = expm(L * t_max) @ rho0_vec
    return jnp.real(rho_final.reshape(5,5)[3,3])  # 3P0 population

# Vectorize over atoms:
simulate_ensemble = jax.vmap(simulate_atom)

# Vectorize over parameter grid:
simulate_grid = jax.vmap(jax.vmap(simulate_atom, in_axes=(0,...)), in_axes=(...,0))
```

**Pros**:
- `jit` + `vmap` gives 100–1000× speedup over the current code
- `jax.grad` gives exact gradients for free → enables gradient-based optimization (L-BFGS, Adam)
- Runs on GPU if available (replace `jax.numpy` → CUDA backend)

**Cons**: Requires rewriting the simulation core in JAX. The erf envelope with time-dependent H requires either discretization or using `jax.experimental.ode` (Dormand-Prince solver).

---

### Strategy E: Precompute Static Operators ⭐⭐

The QuTiP objects `H_01`, `H_12`, `H_23`, `proj`, `c_ops`, `b` are rebuilt on every call to `simulate_three_photon_rabi_dynamics`. These are completely static (independent of all physical parameters). Move them to module-level globals or a cached dataclass so they are constructed once per Python session.

```python
# At module level, precompute once:
_DIM = 5
_B   = [qt.basis(_DIM, i) for i in range(_DIM)]
_H01 = 0.5 * (_B[0]*_B[1].dag() + _B[1]*_B[0].dag())
# etc.
```

**Expected speedup**: 5–15% per call (small but free).

---

## 3. Recommended Path Forward

| Priority | Action | Effort | Speedup |
|----------|--------|--------|---------|
| 1 | Replace QuTiP mesolve with numpy Liouvillian + `solve_ivp` | Medium | 10–50× |
| 2 | Parallelize shots with `joblib` | Low | N_cores× |
| 3 | Batch atoms in vectorized ODE | Medium | 2–5× additional |
| 4 | Switch to JAX for full JIT + grad | High | 100–1000× total |

**Conservative quick win**: Strategies 1 + 2 together can realistically get a 50–100× wall-clock speedup with a few hours of work, reducing a 10-minute run to ~10 seconds.

---

## 4. Parameter Space Optimization

Once the simulation is fast enough (< 1 s per evaluation), the following approaches work well for optimizing over large parameter spaces.

### 4.1 What Are You Optimizing?

Define your objective function clearly. Common targets:

- **Maximize 3P0 population** at a fixed time `t_max`
- **Maximize transfer efficiency** integrated over the pulse (area under 3P0 curve)
- **Robustness**: maximize 3P0 pop while minimizing sensitivity to detuning fluctuations (add finite-difference penalty terms)

### 4.2 Parameter Space Dimensions

Likely free parameters for optimization:

| Parameter | Symbol | Typical range |
|-----------|--------|--------------|
| 689 nm power | `P_689` | 5–50 mW |
| 688 nm power | `P_688` | 5–100 mW |
| 679 nm power | `P_679` | 1–30 mW |
| 689 nm detuning | `delta_0` | ±10 MHz |
| 679 nm detuning | `delta_2` | ±5 MHz (around AC-Stark shifted resonance) |
| Pulse duration | `T_MAX` | 1–20 µs |
| B-field | `B_field_G` | 10–50 G |
| AOM rise time | `sigma_aom` | 50–200 ns |

That's potentially 6–8 free parameters — too many for brute-force grid search (8D grid at 20 points/dim = 25 billion evaluations).

### 4.3 Optimization Methods

#### Bayesian Optimization (Recommended for < 500 evaluations budget)

Uses a Gaussian Process surrogate model to intelligently decide where to sample next, balancing exploration and exploitation. Excellent when each evaluation is expensive (>0.1 s).

```python
# Using optuna (easiest) or scikit-optimize (more control)
import optuna

def objective(trial):
    P_689 = trial.suggest_float('P_689', 5e-3, 50e-3)
    P_688 = trial.suggest_float('P_688', 5e-3, 100e-3)
    delta_2 = trial.suggest_float('delta_2', -5e6, 5e6)
    # ... run simulation ...
    return -pop_3P0_final   # minimize negative population

study = optuna.create_study()
study.optimize(objective, n_trials=200, n_jobs=8)   # parallel trials
print(study.best_params)
```

**Pros**: Works in high dimensions, handles noisy objectives, parallelizes well.
**Cons**: Requires ~50–200 evaluations to converge; GP scales as O(N³) in evaluations.

#### Nelder-Mead / Powell (for low-dimensional, smooth landscapes)

Good for 2–4 parameters when the landscape is smooth. Very simple to implement with `scipy.optimize.minimize`.

```python
from scipy.optimize import minimize

def neg_pop(params):
    P_689, P_688, delta_2 = params
    _, pops = simulate_three_photon_rabi_dynamics(...)
    return -np.max(pops[2])   # maximize 3P0

result = minimize(neg_pop, x0=[19e-3, 11e-3, 0.0], method='Nelder-Mead')
```

#### Differential Evolution (global, gradient-free, parallelizable)

Good for finding global optima in bounded parameter spaces with 5–15 dimensions.

```python
from scipy.optimize import differential_evolution

bounds = [(5e-3, 50e-3), (5e-3, 100e-3), (-5e6*2*PI, 5e6*2*PI)]
result = differential_evolution(neg_pop, bounds, workers=-1, maxiter=500, tol=1e-4)
```

`workers=-1` uses all CPU cores automatically.

#### Gradient-Based (L-BFGS-B) with JAX Autodiff

If you implement Strategy D (JAX), you get exact gradients for free, enabling fast convergence:

```python
import jax
import optax   # or scipy.optimize with jax.value_and_grad

grad_fn = jax.value_and_grad(neg_pop_jax)

# Gradient descent with Adam optimizer:
optimizer = optax.adam(learning_rate=1e-4)
opt_state = optimizer.init(params)

for step in range(1000):
    val, grads = grad_fn(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
```

Gradient-based methods converge in 50–200 iterations vs. 1000s for gradient-free methods.

### 4.4 Parallelizing the Parameter Sweep

For grid scans (e.g., 2D power sweep at fixed detunings), use `joblib`:

```python
from joblib import Parallel, delayed
import itertools

P_689_vals = np.linspace(5e-3, 50e-3, 20)
P_688_vals = np.linspace(5e-3, 100e-3, 20)

def eval_point(P_689, P_688):
    _, pops = simulate_three_photon_rabi_dynamics(
        ..., powers=[P_689, P_688, P_679], ...)
    return np.max(pops[2])

grid = list(itertools.product(P_689_vals, P_688_vals))
results = Parallel(n_jobs=-1)(
    delayed(eval_point)(p0, p1) for p0, p1 in grid
)
pop_grid = np.array(results).reshape(20, 20)
```

### 4.5 Workflow Summary

```
Current bottleneck:
  simulate() takes ~30–120 s  →  can't do optimization at all

Step 1 — Speed up simulation (Strategies A + B + C):
  simulate() takes ~1–5 s     →  Bayesian opt / Nelder-Mead feasible

Step 2 — Run optimization:
  - Start with Bayesian opt (optuna) to map landscape globally
  - Refine with Nelder-Mead or L-BFGS around the best region

Step 3 — Full JAX rewrite (optional but powerful):
  simulate() takes ~0.01–0.1 s → gradient-based opt, thousands of evals,
                                   GPU acceleration, robustness optimization
```

---

## 5. Quick Wins You Can Do Right Now (No Rewrite Required)

1. **Reduce `n_shots` and `N_atoms` during optimization sweeps** — use `n_shots=20, N_atoms=10` to find the landscape, then refine with full parameters at the best point.

2. **Use `SQUARE` mode for optimization** — it's faster than ERF mode (no time-dependent H callbacks) and gives the same resonance/power landscape. Switch to ERF only for final validation.

3. **Precompute `get_calculated_parameters` once** — currently it is called once per shot inside the loop. If you are sweeping detunings (not positions/velocities), those parameters don't change between shots. Cache the result.

4. **Cache the QuTiP static operators** — move `b`, `H_01`, `H_12`, `H_23`, `proj`, `c_ops` out of the function body so they are only built once per session. Add a `functools.lru_cache` or a module-level dict.

5. **Profile first** — run with `n_shots=3, N_atoms=3` and `cProfile` to confirm where the actual time is going before making large changes:
   ```python
   import cProfile
   cProfile.run('simulate_three_photon_rabi_dynamics(...)', sort='cumulative')
   ```
