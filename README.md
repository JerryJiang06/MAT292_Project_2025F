# Hybrid Logistic–Bass Diffusion Model for Steam User Growth

This repository contains a reproducible Python algorithm that fits a **time-varying hybrid Logistic–Bass diffusion ODE** to monthly Steam user counts, using external macro/tech/social driver data(world population, internet penetration, Twitch viewership, tech CPI/PPI). It also generates **future adoption scenarios** by extrapolating the drivers and perturbing fitted parameters.

The primary script is **`new_fit4.py`**, other fit.py are older versions of the algorithm.

---

## 1) Model

Let **\(N(t)\)** be Steam users. Time is measured in **months**.

The fitted ODE is:

$$
\frac{dN}{dt}
= \left[p(t) + q(t)\frac{N(t)}{K(t)}\right] N(t)\left(1-\frac{N(t)}{K(t)}\right).
$$

The coefficient functions are **linear in normalized drivers**:

$$
\begin{aligned}
p(t) &= p_0 + a_1\,\mathrm{CPI}(t) + a_2\,\mathrm{PPI}(t),\\
q(t) &= q_0 + b_1\,\mathrm{Twitch}(t),\\
K(t) &= k_1\,\mathrm{Population}(t) + k_2\,\mathrm{Internet}(t).
\end{aligned}
$$

Fitted Parameters:  
`[p0, a1, a2, q0, b1, k1, k2]`

### optimization
The fitter chooses the parameter vector to minimize the **sum of squared errors** between the simulated curve and observed Steam users over the period:

$$
\min_{\theta}\ \sum_{t}( \hat N_{\theta}(t) - N_{\text{obs}}(t) )^2.
$$

Implementation detail: the residual is `sim - y_obs` and the optimizer minimizes its 2-norm squared.

---

## 2) Data requirements

The script expects **six monthly CSVs** in the same directory as `new_fit4.py`, the repository already holds the csv files used in our research.

Default filenames (as coded):

- `steam_users_04_01.csv` — Steam users (target series)
- `twitch_views_04_01.csv` — Twitch viewership
- `internet_penetration_04_01.csv` — Internet penetration (% of population)
- `world_population_04_01.csv` — World population
- `tech_cpi_04_01.csv` — Technology CPI
- `tech_ppi_04_01.csv` — Technology PPI

**CSV format assumptions**
- One column whose name contains **`date`** (case-insensitive)
- One value column (the first non-date column is used)
- Monthly data (the code aligns everything to month starts)

---

## 3) Installation

Recommended: Python 3.10+

Install dependencies:
```bash
pip install numpy pandas scipy matplotlib scikit-learn python-dateutil
```

## 4) Running the program

By copying the repository to vscode and running 'new_fit4.py', an output folder containing all the plots and fits will be generated. 
MAKE SURE YOU ARE IN THE CORRECT PATH, YOU SHOULD SEE ALL FILES FROM THE REPOSITORY IN YOUR CURRENT FOLDER.


---

## 5) Function Reference for new_fit4.py

This section documents the main functions in `new_fit4.py` in a “role / inputs / outputs / algorithm notes” format.

---

### `load_monthly_csv(filepath, col_name_hint)`

**Role:** Load a CSV and convert it into a standardized monthly time series with columns `["date", <col_name_hint>]`.

**Inputs**
- `filepath` (`str`): Path to a CSV file.
- `col_name_hint` (`str`): Name to assign to the value column (e.g., `"steam"`, `"twitch"`).

**Outputs**
- `pd.DataFrame` with:
  - `date` (`datetime64[ns]`): parsed timestamps
  - `<col_name_hint>` (`float`/numeric): the series values

**Algorithm / Implementation Notes**
- Reads the CSV with `pd.read_csv`.
- Auto-detects the **date column** by selecting the first column whose name contains `"date"` (case-insensitive).
- Selects the **first non-date column** as the value column.
- Cleans date strings (strip whitespace, remove non-ASCII junk).
- Parses dates via `pd.to_datetime(..., errors="coerce")`; raises `ValueError` if any dates become `NaT`.
- Sorts by ascending date .


---

### `create_master_timeline(dfs)`

**Role:** Create a month-start timeline spanning the earliest observed date across all datasets up to `EXTEND_TO_YEAR`.

**Inputs**
- `dfs` (`dict[str, pd.DataFrame]`): each dataframe must contain a `date` column.

**Outputs**
- `pd.DataFrame` with a single `date` column at monthly start frequency (`"MS"`).

**Algorithm / Implementation Notes**
- Start date = minimum of `df["date"].min()` across all datasets.
- End date = `datetime(EXTEND_TO_YEAR, 12, 1)`.
- Uses `pd.date_range(..., freq="MS")`.

---

### `merge_and_fill(master, df, col, fill_strategy="zero_before")`

**Role:** Align one dataset onto the master timeline, interpolate internal missing values, and handle “before the dataset starts” values.

**Inputs**
- `master` (`pd.DataFrame`): must contain `date`.
- `df` (`pd.DataFrame`): must contain `date` and `col`.
- `col` (`str`): value column name to fill (e.g., `"steam"`, `"internet"`).
- `fill_strategy` (`str`):
  - `"zero_before"`: fill months earlier than the first observation with `0.0`
  - `"hold_before"`: fill months earlier than the first observation with the first observed value
  - `"nan_before"`: leave months earlier than the first observation as NaN

**Outputs**
- `pd.DataFrame`: `master` merged with `col`.

**Algorithm / Implementation Notes**
- Left-merge: `master.merge(df, on="date", how="left")`.
- Fills internal gaps via linear interpolation.
- Applies the chosen pre-start strategy for months before the dataset begins.

---

### `extrapolate_driver(series, months_out, method)`

**Role:** Driver-specific extrapolation wrapper that selects an extrapolation.

**Inputs**
- `series` (`pd.Series`): datetime index, numeric values.
- `months_out` (`int`): number of months to extend forward.
- `method` (`str`): one of `{ "linear", "exp", "logistic" }`.

**Outputs**
- `pd.Series`: original series with future months appended.

**Algorithm / Implementation Notes**
- Preprocess: drop NaNs, ensure datetime index, sort.
- Dispatch:
  - `"linear"` → `extrapolate_future(..., method="linear")`
  - `"exp"` → `extrapolate_future(..., method="exp")`
  - `"logistic"` → capped growth heuristic:
    - set `K = 1.1 * max(series)`
    - fit a line to the logit transform over the last ~36 points
    - generate future values with a logistic curve
  - checks if logistic is unsafe, if so, falls back to linear.

---

### `extrapolate_future(series, months_out, method, fit_window=FIT_WINDOW_MONTHS)`

**Role:** Extrapolation routine, implements **linear** or **exponential** trend extension using a recent fitting window.

**Inputs**
- `series` (`pd.Series`): datetime index.
- `months_out` (`int`)
- `method` (`str`): `"linear"` or `"exp"`
- `fit_window` (`int`): number of most-recent months used for fitting.

**Outputs**
- `pd.Series`: original + extrapolated future values.

**Algorithm / Implementation Notes**
- Uses the last `fit_window` points if available (else all points).
- Linear: fit `y ~ t` and extrapolate.
- Exponential: fit `log(y) ~ t` and exponentiate predictions.
  - if any `y <= 0`, exponential is invalid and falls back to linear.
- Future index is month-starts beginning next month.

---

### `split_historical_future(df, cutoff_year)`

**Role:** Split the merged master dataframe into historical (fit) and future (projection) partitions.

**Inputs**
- `df` (`pd.DataFrame`): must contain `date`.
- `cutoff_year` (`int`): e.g., `FIT_END_YEAR = 2025`.

**Outputs**
- `(hist_df, future_df)` as two `pd.DataFrame`s.

**Algorithm / Implementation Notes**
- `hist_df`: rows where `date.year <= cutoff_year`
- `future_df`: rows where `date.year > cutoff_year`

---

### `build_future_parameters(fitted_params, hist_df, future_df)`

**Role:** Convert the fitted parameter vector into **future monthly coefficient arrays** \(p(t), q(t), K(t)\) over the projection window.

**Inputs**
- `fitted_params` (`array-like`): `[p0, a1, a2, q0, b1, k1, k2]`
- `hist_df`, `future_df` (`pd.DataFrame`): must include normalized driver columns:
  - `cpi_norm`, `ppi_norm`, `twitch_norm`, `pop_norm`, `internet_norm`

**Outputs**
- `(p_t, q_t, K_t)` as NumPy arrays for the **future** months.

**Equations**
- `p(t) = p0 + a1*CPI(t) + a2*PPI(t)`
- `q(t) = q0 + b1*Twitch(t)`
- `K(t) = k1*Population(t) + k2*Internet(t)`

**Notes**
- Clipping/guards are applied later (during residual computation), not here.

---

### `simulate_hybrid(p_t, q_t, K_t, N0, t_len)`

**Role:** Numerically integrate the hybrid Logistic–Bass ODE to produce a simulated Steam-user curve.

**Inputs**
- `p_t, q_t, K_t` (`np.ndarray`): monthly coefficient arrays (length ≥ `t_len`).
- `N0` (`float`): initial condition \(N(0)\).
- `t_len` (`int`): number of months to simulate.

**Outputs**
- `np.ndarray` of length `t_len`: simulated \(N(t)\) evaluated at integer months.

**Model**

$$
\frac{dN}{dt}=\left(p(t)+q(t)\frac{N}{K(t)}\right)\,N\left(1-\frac{N}{K(t)}\right)
$$

**Algorithm / Implementation Notes**
- Uses `scipy.integrate.solve_ivp` with `method="RK45"`.
- Coefficients are treated as **piecewise-constant** by month (continuous time mapped to `floor(t)`).
- Guards against numerical issues (e.g., division by tiny `K`, overshooting `K`).
- Evaluates solution at month indices `t = 0,1,...,t_len-1`.
- If solver fails, returns a large penalty array so the optimizer rejects this parameter set.

---

### `residuals_for_params(x, drivers, y_obs, N0)`

**Role:** Convert a candidate parameter vector into a residual vector used by least squares.

**Inputs**
- `x` (`np.ndarray`): `[p0, a1, a2, q0, b1, k1, k2]`
- `drivers` (dict-like / dataframe-like): must provide arrays for:
  - `internet_norm`, `pop_norm`, `twitch_norm`
  - `cpi_norm`, `ppi_norm` (if missing, treated as zeros)
- `y_obs` (`np.ndarray`): observed Steam users.
- `N0` (`float`): initial condition.

**Outputs**
- `np.ndarray`: residuals `sim - y_obs` (same length as `y_obs`).

**Algorithm / Implementation Notes**
1. Build coefficient arrays `p_t`, `q_t`, `K_t` from `x` and drivers.
2. Apply safety constraints (typical):
   - enforce `p_t >= small_positive`
   - enforce `q_t >= 0`
   - enforce `K_t` stays above the observed max (so capacity does not undercut data)
3. If any invalid values (NaN/inf/nonpositive capacity), return a large constant residual vector.
4. Simulate via `simulate_hybrid` and return `sim - y_obs`.

---

### `fit_parameters_auto(drivers, y_obs, N0)`

**Role:** Fit parameters by minimizing the **sum of squared errors** between simulation and observations.

**Inputs**
- `drivers`: the normalized driver arrays used by `residuals_for_params`.
- `y_obs` (`np.ndarray`)
- `N0` (`float`)

**Outputs**
- `(best_x, result)` where:
  - `best_x`: fitted parameter vector
  - `result`: SciPy optimizer result from the local refinement step

**Objective**

$$
\min_{\theta}\ \sum_{t=0}^{T-1}\left(N_{\text{sim}}(t;\theta)-N_{\text{obs}}(t)\right)^2
$$

**Algorithm / Implementation Notes**
- Uses a two-stage approach:
  1. **Global**: `differential_evolution` on SSE
  2. **Local**: `least_squares` refinement on residual vector
- Bounds are enforced throughout to keep parameters in a physically/numerically reasonable region.

---

### `run_future_scenarios(fitted_params, hist_df, future_df, N0, scenarios)`

**Role:** Run “what-if” projections by perturbing fitted parameters and re-simulating into the future.

**Inputs**
- `fitted_params` (`np.ndarray`)
- `hist_df`, `future_df` (`pd.DataFrame`): must include normalized drivers.
- `N0` (`float`)
- `scenarios` (`dict`): scenario name → perturbation spec.

**Outputs**
- `dict[str, np.ndarray]`: scenario name → simulated curve.
- Writes per-scenario CSVs to `output_results/scenario_<name>.csv`.

**Scenario spec format**
```python
scenarios = {
  "baseline": {},
  "high_social": {"params": {"b1": ("mult", 1.5)}},
  "low_macro":  {"params": {"a1": ("mult", 0.7), "a2": ("mult", 0.7)}}
}

