//! Interpolation and profile-conversion utilities for second-order path parameterization.
//!
//! # 模块职责与调用流程
//!
//! 本模块提供「求解器输出 a(s)」→「时间域轨迹 s(t)」的完整转换链。
//!
//! ## 转换链（2阶求解器后处理）
//! ```text
//! topp2_ra() / copp2_socp()
//!   → a: Vec<f64>        (a[k] = ṡ² at s[k], 节点值)
//!   ↓
//! a_to_b_topp2(s, &a)
//!   → b: Vec<f64>        (b[k] = (a[k+1]-a[k])/(2Δs), 段值，共 n-1 个)
//!   ↓
//! s_to_t_topp2(s, &a, t0)
//!   → (t_final, t_s: Vec<f64>)   (t_s[k] = t(s[k])，累积时间)
//!   ↓
//! t_to_s_topp2(s, &a, &t_s, InterpolationMode::UniformTimeGrid(0.0, dt, true))
//!   → s_t: Vec<f64>      (均匀时间采样的 s(t) 值，用于关节空间重建)
//! ```
//!
//! ## 与 plot_joint_trajectory.py 的对应关系
//! s_t 即脚本中的 `traj["s_t"]`，再通过 np.interp 映射回 q/dq/ddq。
//!
//! # Method identity
//! This module serves both:
//! - **Time-Optimal Path Parameterization (TOPP2)** workflows,
//! - **Convex-Objective Path Parameterization (COPP2)** workflows.
//!
//! # Conventions
//! - Path grid uses station samples `s[0..=n]`.
//! - State profile `a` is node-based (`a.len() == s.len()`).
//! - Profile `b` is segment-based (`b.len() == s.len() - 1`).

use crate::copp::InterpolationMode;
use itertools::izip;

/// Compute segment profile `b` from node profile `a`.
///
/// # Definition
/// For each segment `[s_k, s_{k+1}]`, this function computes:
/// $b_k = \frac{1}{2}\frac{a_{k+1}-a_k}{s_{k+1}-s_k}$.
///
/// # Input contract
/// - valid when `s.len() >= 2` and `a.len() == s.len()`;
/// - otherwise returns an empty vector.
///
/// # Returns
/// Returns `b` with `b.len() == s.len() - 1`.
///
/// # Errors
/// This function does not return `Result`; invalid input is mapped to empty output.
///
/// # Contract
/// - Output ordering is consistent with segment ordering on `s.windows(2)`.
/// - No allocation beyond returned vector and iterator temporaries.
pub fn a_to_b_topp2(s: &[f64], a: &[f64]) -> Vec<f64> {
    if s.len() < 2 || a.len() != s.len() {
        return vec![];
    }
    s.windows(2)
        .zip(a.windows(2))
        .map(|(s_pair, a_pair)| 0.5 * (a_pair[1] - a_pair[0]) / (s_pair[1] - s_pair[0]))
        .collect::<Vec<f64>>()
}

/// Compute cumulative time profile `t(s)` from `a(s)`.
///
/// # Semantics
/// - `t_s[i]` is the time at station `s[i]`.
/// - initial condition is `t_s[0] = t0`.
/// - returns `(t_final, t_s)` where `t_final == *t_s.last().unwrap()`.
///
/// # Input contract
/// - valid when `s.len() >= 2` and `a.len() == s.len()`;
/// - otherwise returns `(NaN, empty)`.
///
/// # Returns
/// Returns `(t_final, t_s)` with `t_s.len() == s.len()` on valid input.
///
/// # Errors
/// This function does not return `Result`; invalid input is mapped to `(NaN, vec![])`.
///
/// # Contract
/// - `t_s` is monotonically increasing when `a` is nonnegative and `s` is increasing.
/// - `t_s[0] == t0` always holds on valid input.
pub fn s_to_t_topp2(s: &[f64], a: &[f64], t0: f64) -> (f64, Vec<f64>) {
    if s.len() < 2 || a.len() != s.len() {
        return (f64::NAN, vec![]);
    }
    // Trapezoidal integration of dt = ds/√a(s):
    //   Δt_k = ∫_{s_k}^{s_{k+1}} ds/√a(s)
    //        ≈ (s_{k+1} - s_k) / ((√a_k + √a_{k+1}) / 2)   [trapezoidal in 1/√a]
    //        = 2·(s_{k+1} - s_k) / (√a_k + √a_{k+1})
    // This matches the TOPP2 time parameterization exactly when a(s) is piecewise-linear.
    let mut t_s = Vec::<f64>::with_capacity(s.len()); // t_s[i] = t(s[i]), begin from t0
    let mut t_prev = t0;
    t_s.push(t_prev);
    for (s_pair, a_pair) in s.windows(2).zip(a.windows(2)) {
        t_prev += 2.0 * (s_pair[1] - s_pair[0]) / (a_pair[0].sqrt() + a_pair[1].sqrt());
        t_s.push(t_prev);
    }
    (t_prev, t_s)
}

/// Interpolate inverse mapping `s(t)` from `a(s)` and sampled `t(s)`.
///
/// # Modes
/// - `UniformTimeGrid(t0, dt, include_final)`: generate uniform time samples;
/// - `NonUniformTimeGrid(t_sample)`: use caller-provided increasing samples.
///
/// # Input contract
/// - requires `s.len() >= 2`, `a.len() == s.len()`, `t_s.len() == s.len()`;
/// - requires `t_s` strictly increasing.
/// - invalid input returns empty vector.
///
/// # Output semantics
/// - output length matches requested sample count in each mode;
/// - for out-of-range time samples, output entries are `NaN`.
///
/// # Returns
/// Returns sampled `s(t)` values according to `mode`.
///
/// # Errors
/// This function does not return `Result`; invalid input or invalid `mode` settings
/// are mapped to empty output.
///
/// # Contract
/// - preserves requested sample order;
/// - never panics for malformed user input (falls back to empty vector).
pub fn t_to_s_topp2(s: &[f64], a: &[f64], t_s: &[f64], mode: InterpolationMode<'_>) -> Vec<f64> {
    if s.len() < 2
        || a.len() != s.len()
        || t_s.len() != s.len()
        || t_s.windows(2).any(|w| w[0] >= w[1])
    {
        return vec![];
    }
    match mode {
        InterpolationMode::UniformTimeGrid(t0, dt, include_final) => {
            if dt <= 0.0 {
                return vec![];
            }
            // num_t * dt + t0 <= t_final
            let num_t = ((t_s.last().unwrap() - t0) / dt).floor() as usize;
            let mut s_t =
                t_to_s_topp2_core(s, a, t_s, (0..num_t).map(|i| t0 + i as f64 * dt), num_t);
            if include_final {
                let flag = if s_t.is_empty() {
                    t0 <= *t_s.last().unwrap()
                } else {
                    *s_t.last().unwrap() < *s.last().unwrap()
                };
                if flag {
                    s_t.push(*s.last().unwrap());
                }
            }
            s_t
        }
        InterpolationMode::NonUniformTimeGrid(t_sample) => {
            if t_sample.is_empty() || t_sample.windows(2).any(|w| w[0] >= w[1]) {
                // Empty or non-increasing sample sequence is invalid.
                return vec![];
            }
            t_to_s_topp2_core(s, a, t_s, t_sample.iter().cloned(), t_sample.len())
        }
    }
}

/// Core inverse interpolation kernel for `t_to_s_topp2`.
///
/// It consumes increasing `t_sample` values and emits corresponding `s(t)` by
/// segment-wise inversion with quadratic-in-`a` local model.
fn t_to_s_topp2_core(
    s: &[f64],
    a: &[f64],
    t_s: &[f64],
    mut t_sample: impl Iterator<Item = f64>,
    len_t_sample: usize,
) -> Vec<f64> {
    let &t_start = t_s.first().unwrap();
    // Map t to s
    let mut s_t = Vec::<f64>::with_capacity(len_t_sample + 1); // s_t[i] = s(t[i])
    let Some(mut t_curr) = t_sample.next() else {
        return vec![];
    };
    while t_curr < t_start {
        s_t.push(f64::NAN);
        let Some(t) = t_sample.next() else {
            return s_t;
        };
        t_curr = t;
    }

    for (s_pair, a_pair, t_pair) in izip!(s.windows(2), a.windows(2), t_s.windows(2)) {
        while t_curr <= t_pair[1] {
            s_t.push(
                s_pair[0]
                    + inverse_2order(
                        a_pair[0],
                        (a_pair[1] - a_pair[0]) / (s_pair[1] - s_pair[0]),
                        0.0,
                        t_curr - t_pair[0],
                    ),
            );
            let Some(t) = t_sample.next() else {
                return s_t;
            };
            t_curr = t;
        }
    }

    s_t.push(f64::NAN);
    while t_sample.next().is_some() {
        s_t.push(f64::NAN);
    }
    s_t
}

/// Solve `x_right` from the integral equation
/// $dt = \int_{x_{left}}^{x_{right}} \frac{dx}{\sqrt{c_0 + c_1 x}}$.
///
/// Within a segment [s_k, s_{k+1}], a(s) is modeled as linear:
///   a(s) = a_k + (a_{k+1} - a_k) / (s_{k+1} - s_k) * (s - s_k)
///        = c0 + c1 * (s - s_k)
/// so ds/dt = sqrt(a) = sqrt(c0 + c1*x), giving dt = dx/sqrt(c0+c1*x).
///
/// Closed-form inversion:
///   If c1 ≠ 0: integral = 2/c1 * [sqrt(c0+c1*x_right) - sqrt(c0+c1*x_left)] = dt
///              => sqrt(c0+c1*x_right) = sqrt(c0+c1*x_left) + c1*dt/2
///              => x_right = [(sqrt(c0+c1*x_left) + c1*dt/2)² - c0] / c1
///   If c1 = 0 (constant speed segment): dt = dx/sqrt(c0) => x_right = x_left + sqrt(c0)*dt
///   If both zero (stopped): x_right = infinity (singularity / zero-speed degenerate)
#[inline]
fn inverse_2order(c0: f64, c1: f64, x_left: f64, dt: f64) -> f64 {
    if dt == 0.0 {
        x_left
    } else if c1.abs() > f64::EPSILON {
        // Linear a(s): closed-form quadratic inversion
        (((c0 + c1 * x_left).sqrt() + 0.5 * c1 * dt).powi(2) - c0) / c1
    } else if c0.abs() > f64::EPSILON {
        // Constant a(s): uniform path speed, s advances as sqrt(a)*dt
        x_left + c0.sqrt() * dt
    } else {
        // a = 0 everywhere on this segment: robot is stopped, time is infinite
        f64::INFINITY
    }
}
