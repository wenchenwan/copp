//! Objective definitions and shared validators for COPP2/COPP3.
//!
//! # 目标函数在系统中的位置
//!
//! 目标函数由 `CoppObjective` 枚举定义，只在 COPP 求解器（非 TOPP）中使用：
//! ```text
//! 用户定义 objectives = [CoppObjective::Time(1.0), CoppObjective::ThermalEnergy(0.1, &w)]
//!   ↓
//! Copp2ProblemBuilder::new(&robot, ..., &objectives).build()
//!   ↓  validate_copp2_objectives()   检查权重非负、维度匹配
//!   ↓
//! Copp2Problem { objectives: &objectives, ... }
//!   ↓
//! copp2_socp() / copp3_socp()
//!   └─ 将 objectives 转化为 Clarabel 的目标函数系数向量 q
//!       Time:         q_k = w_t / sqrt(a_lin[k])    (对 a[k] 的线性化时间代价)
//!       ThermalEnergy:q_k = w_e * sum_i(τ_i·ν_i)²  (力矩加权热能代价)
//!       Linear:       q_k = w_l * alpha[k]            (用户自定义线性项)
//! ```
//!
//! TOPP 求解器（topp2_ra / topp3_lp / topp3_socp）不接受 objectives，
//! 隐式使用时间最优目标（贪心最大化 a[k]）。
//!
//! # Basic symbols (consistent with constraints)
//! - $a(s)=\dot s^2$, sampled as `a[k]=a(s_k)`.
//! - $b(s)=\ddot s$.
//! - Path grid is `s[0], s[1], ..., s[n-1]` with `n = s.len()`.
//!
//! Discretization difference:  
//! - COPP2:  
//!   - `a.len() == s.len() == n`  
//!   - `b.len() == n-1`, where `b[k]` is on interval `[s_k, s_{k+1}]`  
//!   - typically `b[k] = (a[k+1]-a[k]) / (2*(s[k+1]-s[k]))`  
//! - COPP3:  
//!   - `a.len() == b.len() == s.len() == n`  
//!   - `b[k] = b(s_k)` is node-based.

use crate::diag::CoppError;

/// Objective terms for COPP optimization.  
/// Continuous formulation is shared by COPP2/COPP3; discrete form depends on how `b` and torque are sampled.  
/// Torque notation:  
/// - continuous: $\boldsymbol{\tau}(s)$.  
/// - discrete: `tau[i][k]` for joint `i` at `s[k]`.  
/// - COPP2: `tau[i][k]` is the right-limit value $\boldsymbol{\tau}(s_k^+)$, i.e. computed on interval $[s_k, s_{k+1}]$ from `(a[k], a[k+1], b[k])`.  
/// - COPP3: `tau[i][k]` is node value $\boldsymbol{\tau}(s_k)$, computed from `(a[k], b[k])`.
pub enum CoppObjective<'a> {
    /// 时间目标：最小化总运动时间。
    ///
    /// 权重 `w_t > 0`，值越大越重视时间最优。
    /// 连续形式：J = w_t · ∫ ds/√a(s)
    /// COPP2 离散化：J = 2w_t · Σ (s[k+1]-s[k]) / (√a[k]+√a[k+1])  （梯形积分）
    /// COPP3 离散化：J = w_t · Σ weight_a_time[k] / √a[k]
    Time(f64),

    /// 热能目标：最小化电机发热（力矩平方×时间积分）。
    ///
    /// 参数：`(w_e, normalize)`，其中 `normalize[i]` 为第 i 轴的归一化系数 ν_i。
    /// 连续形式：J = w_e · ∫ Σ_i (τ_i·ν_i)² / √a(s) ds
    /// 适用场景：在时间最优基础上减少电机热损耗（COPP3 demo 默认组合目标之一）。
    /// 注意：该目标需要 RobotTorque（逆动力学），仅在 COPP2/COPP3 中有效。
    ThermalEnergy(f64, &'a [f64]),

    /// 力矩全变差目标：最小化力矩变化幅度，使运动更平滑。
    ///
    /// 连续形式：J = w_v · Σ_i ∫ |dτ_i/ds| · ν_i ds
    /// 离散化：J = w_v · Σ_i Σ_k |τ_i[k+1]-τ_i[k]| · ν_i
    /// 适用场景：抑制力矩突变，降低机械冲击。
    TotalVariationTorque(f64, &'a [f64]),

    /// 用户自定义线性目标：对 a[k] 和 b[k] 的加权求和。
    ///
    /// 参数：`(w_l, alpha, beta)`
    /// 连续形式：J = w_l · ∫ (α(s)·a(s) + β(s)·b(s)) ds
    /// COPP2：alpha.len()==n, beta.len()==n-1
    /// COPP3：alpha.len()==beta.len()==n
    /// 适用场景：嵌入自定义代价（如能量代理、位置偏差等）。
    Linear(f64, &'a [f64], &'a [f64]),
}

/// Validate objective terms for COPP2.  
/// `s_len` must be the number of grid points in the closed interval `[idx_s_start, idx_s_final]`, i.e. `idx_s_final - idx_s_start + 1`.  
/// Rules:  
/// - `objectives` must be non-empty.  
/// - all weights must be non-negative and finite.  
/// - for `ThermalEnergy` / `TotalVariationTorque`, `normalize.len() == dim`.  
/// - all normalize entries must be non-negative and finite.  
/// - for `Linear`, `alpha.len() == s_len` and `beta.len() + 1 == s_len`.
#[inline]
pub(crate) fn validate_copp2_objectives(
    function_name: &str,
    objectives: &[CoppObjective<'_>],
    dim: usize,
    s_len: usize,
) -> Result<(), CoppError> {
    if objectives.is_empty() {
        return Err(CoppError::InvalidInput(
            function_name.into(),
            "objectives must not be empty".into(),
        ));
    }

    for (i, objective) in objectives.iter().enumerate() {
        match objective {
            CoppObjective::Time(weight) => {
                if weight.is_nan() || weight.is_infinite() || *weight < 0.0 {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!(
                            "objectives[{i}]::Time(weight={weight}) must be finite and non-negative"
                        ),
                    ));
                }
            }
            CoppObjective::ThermalEnergy(weight, normalize)
            | CoppObjective::TotalVariationTorque(weight, normalize) => {
                if weight.is_nan() || weight.is_infinite() || *weight < 0.0 {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!("objectives[{i}] weight={weight} must be finite and non-negative"),
                    ));
                }
                if normalize.len() != dim {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!(
                            "objectives[{i}] normalize.len()={} must equal robot dim={dim}",
                            normalize.len()
                        ),
                    ));
                }
                for (j, v) in normalize.iter().enumerate() {
                    if v.is_nan() || v.is_infinite() || *v < 0.0 {
                        return Err(CoppError::InvalidInput(
                            function_name.into(),
                            format!(
                                "objectives[{i}] normalize[{j}]={v} must be finite and non-negative"
                            ),
                        ));
                    }
                }
            }
            CoppObjective::Linear(weight, alpha, beta) => {
                if weight.is_nan() || weight.is_infinite() || *weight < 0.0 {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!(
                            "objectives[{i}]::Linear(weight={weight}) must be finite and non-negative"
                        ),
                    ));
                }
                if alpha.len() != s_len || beta.len() + 1 != s_len {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!(
                            "objectives[{i}]::Linear length mismatch: alpha.len()={}, beta.len()={}, expected alpha.len()={s_len}, beta.len()={}",
                            alpha.len(),
                            beta.len(),
                            s_len - 1
                        ),
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Validate objective terms for COPP3.  
/// `s_len` must be the number of grid points used by TOPP3/COPP3 (typically `a_linear.len()`).  
/// Rules:  
/// - `objectives` must be non-empty.  
/// - all weights must be non-negative and finite.  
/// - for `ThermalEnergy` / `TotalVariationTorque`, `normalize.len() == dim`.  
/// - all normalize entries must be non-negative and finite.  
/// - for `Linear`, `alpha.len() == s_len` and `beta.len() == s_len`.
#[inline]
pub(crate) fn validate_copp3_objectives(
    function_name: &str,
    objectives: &[CoppObjective<'_>],
    dim: usize,
    s_len: usize,
) -> Result<(), CoppError> {
    if objectives.is_empty() {
        return Err(CoppError::InvalidInput(
            function_name.into(),
            "objectives must not be empty".into(),
        ));
    }

    for (i, objective) in objectives.iter().enumerate() {
        match objective {
            CoppObjective::Time(weight) => {
                if weight.is_nan() || weight.is_infinite() || *weight < 0.0 {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!(
                            "objectives[{i}]::Time(weight={weight}) must be finite and non-negative"
                        ),
                    ));
                }
            }
            CoppObjective::ThermalEnergy(weight, normalize)
            | CoppObjective::TotalVariationTorque(weight, normalize) => {
                if weight.is_nan() || weight.is_infinite() || *weight < 0.0 {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!("objectives[{i}] weight={weight} must be finite and non-negative"),
                    ));
                }
                if normalize.len() != dim {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!(
                            "objectives[{i}] normalize.len()={} must equal robot dim={dim}",
                            normalize.len()
                        ),
                    ));
                }
                for (j, v) in normalize.iter().enumerate() {
                    if v.is_nan() || v.is_infinite() || *v < 0.0 {
                        return Err(CoppError::InvalidInput(
                            function_name.into(),
                            format!(
                                "objectives[{i}] normalize[{j}]={v} must be finite and non-negative"
                            ),
                        ));
                    }
                }
            }
            CoppObjective::Linear(weight, alpha, beta) => {
                if weight.is_nan() || weight.is_infinite() || *weight < 0.0 {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!(
                            "objectives[{i}]::Linear(weight={weight}) must be finite and non-negative"
                        ),
                    ));
                }
                if alpha.len() != s_len || beta.len() != s_len {
                    return Err(CoppError::InvalidInput(
                        function_name.into(),
                        format!(
                            "objectives[{i}]::Linear length mismatch: alpha.len()={}, beta.len()={}, expected alpha.len()={s_len}, beta.len()={s_len}",
                            alpha.len(),
                            beta.len(),
                        ),
                    ));
                }
            }
        }
    }

    Ok(())
}
