//! Small numerical helpers shared by LP and geometric kernels.
//!
//! # 模块功能
//!
//! 本模块提供无分配、纯计算的代数基础工具，被 `lp.rs` 和约束组装代码使用：
//! - `cross_product_2d(x1, x2)` — 2D 叉积，结果 > 0 表示 x2 相对 x1 为逆时针方向
//! - `solve_2x2(A, b)` — 2×2 线性系统求解，用于 `force_positive_a` 中的参数计算
//!
//! # Method identity
//! This module provides deterministic algebraic primitives with no allocation:
//! - cross products in 2D/3D,
//! - convexity predicates for 1D/2D affine supports.

/// 2D cross product under right-hand rule.  
/// If result>0, x2 is **anticlockwise** to x1.
#[inline]
pub(crate) fn cross_product_2d(x1: (f64, f64), x2: (f64, f64)) -> f64 {
    x1.0 * x2.1 - x1.1 * x2.0
}
