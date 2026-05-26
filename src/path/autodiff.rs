//! Third-order forward-mode automatic differentiation primitives.
//!
//! `Jet3` carries value and derivatives up to the 3rd order with respect
//! to a single scalar seed variable, and supports common arithmetic plus
//! elementary functions (`sin`, `cos`, `exp`, `ln`, `sqrt`, `powi`).
//!
//! # 调用流程
//! ```text
//! Path::from_parametric(|s: Jet3| vec![sin(s), cos(s)], 0.0, 1.0)
//!   └─ 用户闭包接收 Jet3::seed(s_val)          ← 以 s 为自变量构造种子
//!        └─ 闭包中所有算术运算自动传播导数      ← 运算符重载 (Add/Sub/Mul/Div)
//!             └─ 闭包返回 Vec<Jet3>             ← 每个分量携带 (v, d1, d2, d3)
//!                  └─ path_core::eval_parametric 从 jet.v / d1 / d2 / d3
//!                     分别写入 q / dq / ddq / dddq 输出矩阵
//! ```
//!
//! 整个过程无需用户手动求导：只需写出 q(s) 的解析表达式，
//! 三阶导数 dq/ds、d²q/ds²、d³q/ds³ 自动计算完毕。

use std::ops::{Add, Div, Mul, Neg, Sub};

/// 三阶前向自动微分标量，表示函数 f(s) 在某点 s 处的值和前三阶导数。
///
/// # 字段含义
/// - `v`  : 函数值 f(s)
/// - `d1` : 一阶导数 f'(s) = df/ds
/// - `d2` : 二阶导数 f''(s) = d²f/ds²
/// - `d3` : 三阶导数 f'''(s) = d³f/ds³
///
/// # 使用方式
/// 通过 `Jet3::seed(s)` 创建自变量，然后用普通 Rust 算术表达式
/// 组合出 q(s) 的符号形式；加减乘除及各基本函数均已重载，
/// 导数按链式法则自动传播。
#[derive(Clone, Copy, Debug, Default)]
pub struct Jet3 {
    pub v: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
}

impl Jet3 {
    /// 构造常数：函数值为 v，所有导数为 0。
    /// 用于在路径闭包中表示数值常量（如频率、振幅等参数）。
    #[inline(always)]
    pub fn constant(v: f64) -> Self {
        Self {
            v,
            d1: 0.0,
            d2: 0.0,
            d3: 0.0,
        }
    }

    /// 构造自变量种子：f(s) = s，则 f'=1, f''=0, f'''=0。
    ///
    /// 调用方：`path_core::eval_parametric` 对每个采样点 s_val 调用
    /// `eval_fn(Jet3::seed(s_val))`，将路径函数作用于该种子，
    /// 导数随后在闭包内部的算术运算中自动前向传播。
    #[inline(always)]
    pub fn seed(v: f64) -> Self {
        Self {
            v,
            d1: 1.0,
            d2: 0.0,
            d3: 0.0,
        }
    }

    #[inline(always)]
    pub fn sin(self) -> Self {
        // f = sin(u),  f' = cos(u)·u',  f'' = -sin(u)·u'² + cos(u)·u'',
        // f''' = -cos(u)·u'³ - 3sin(u)·u'·u'' + cos(u)·u'''
        let sv = self.v.sin();
        let cv = self.v.cos();
        let d1sq = self.d1 * self.d1;
        let d1 = cv * self.d1;
        let d2 = -sv * d1sq + cv * self.d2;
        let d3 = -cv * d1sq * self.d1 - 3.0 * sv * self.d1 * self.d2 + cv * self.d3;
        Self { v: sv, d1, d2, d3 }
    }

    #[inline(always)]
    pub fn cos(self) -> Self {
        // f = cos(u),  f' = -sin(u)·u',  f'' = -cos(u)·u'² - sin(u)·u'',
        // f''' = sin(u)·u'³ - 3cos(u)·u'·u'' - sin(u)·u'''
        let sv = self.v.sin();
        let cv = self.v.cos();
        let d1sq = self.d1 * self.d1;
        let d1 = -sv * self.d1;
        let d2 = -cv * d1sq - sv * self.d2;
        let d3 = sv * d1sq * self.d1 - 3.0 * cv * self.d1 * self.d2 - sv * self.d3;
        Self { v: cv, d1, d2, d3 }
    }

    #[inline(always)]
    pub fn exp(self) -> Self {
        // f = exp(u),  f' = exp(u)·u',  f'' = exp(u)·(u'² + u''),
        // f''' = exp(u)·(u'³ + 3u'·u'' + u''')
        let ev = self.v.exp();
        let d1sq = self.d1 * self.d1;
        let d1 = ev * self.d1;
        let d2 = ev * (d1sq + self.d2);
        let d3 = ev * (d1sq * self.d1 + 3.0 * self.d1 * self.d2 + self.d3);
        Self { v: ev, d1, d2, d3 }
    }

    #[inline(always)]
    pub fn ln(self) -> Self {
        // f = ln(u),  f' = u'/u,  f'' = -u'²/u² + u''/u,
        // f''' = 2u'³/u³ - 3u'·u''/u² + u'''/u
        let v = self.v.ln();
        let inv_x = 1.0 / self.v;
        let inv_x2 = inv_x * inv_x;
        let inv_x3 = inv_x2 * inv_x;
        let d1sq = self.d1 * self.d1;
        let d1 = inv_x * self.d1;
        let d2 = -inv_x2 * d1sq + inv_x * self.d2;
        let d3 = 2.0 * inv_x3 * d1sq * self.d1 - 3.0 * inv_x2 * self.d1 * self.d2 + inv_x * self.d3;
        Self { v, d1, d2, d3 }
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        // f = sqrt(u) = u^(1/2),  f' = u'/(2√u),
        // f'' = -u'²/(4u^(3/2)) + u''/(2√u),
        // f''' = 3u'³/(8u^(5/2)) - 3u'·u''/(4u^(3/2)) + u'''/(2√u)
        let sqrtv = self.v.sqrt();
        let inv_sqrt = 1.0 / sqrtv;           // 1/√u
        let inv_v_sqrt = inv_sqrt / self.v;    // 1/u^(3/2)
        let inv_v2_sqrt = inv_v_sqrt / self.v; // 1/u^(5/2)
        let d1sq = self.d1 * self.d1;
        let d1 = 0.5 * inv_sqrt * self.d1;
        let d2 = -0.25 * inv_v_sqrt * d1sq + 0.5 * inv_sqrt * self.d2;
        let d3 = 0.375 * inv_v2_sqrt * d1sq * self.d1 - 0.75 * inv_v_sqrt * self.d1 * self.d2
            + 0.5 * inv_sqrt * self.d3;
        Self {
            v: sqrtv,
            d1,
            d2,
            d3,
        }
    }

    #[inline(always)]
    pub fn powi(self, n: i32) -> Self {
        // f = u^n, using Faà di Bruno's formula for composed power:
        // f'   = n·u^(n-1)·u'
        // f''  = n(n-1)·u^(n-2)·u'² + n·u^(n-1)·u''
        // f''' = n(n-1)(n-2)·u^(n-3)·u'³ + 3n(n-1)·u^(n-2)·u'·u'' + n·u^(n-1)·u'''
        if n == 0 {
            return Self::constant(1.0);
        }
        let nf = n as f64;
        // Compute v^(n-3) once; derive v^(n-2), v^(n-1), v^n by repeated multiplication.
        // This replaces 4 independent `f64::powi` calls with 1 call + 3 multiplications,
        // avoiding redundant repeated-squaring work for the same base.
        let vn3 = self.v.powi(n - 3);
        let vn2 = vn3 * self.v;
        let vn1 = vn2 * self.v;
        let vn = vn1 * self.v;
        let dv = nf * vn1;                             // n·u^(n-1)
        let ddv = nf * (nf - 1.0) * vn2;              // n(n-1)·u^(n-2)
        let dddv = nf * (nf - 1.0) * (nf - 2.0) * vn3; // n(n-1)(n-2)·u^(n-3)
        let d1sq = self.d1 * self.d1;
        let d1 = dv * self.d1;
        let d2 = ddv * d1sq + dv * self.d2;
        let d3 = dddv * d1sq * self.d1 + 3.0 * ddv * self.d1 * self.d2 + dv * self.d3;
        Self { v: vn, d1, d2, d3 }
    }

    #[inline(always)]
    fn inv(self) -> Self {
        // f = 1/u,  f' = -u'/u²,  f'' = 2u'²/u³ - u''/u²,
        // f''' = -6u'³/u⁴ + 6u'·u''/u³ - u'''/u²
        let v = 1.0 / self.v;
        let inv_x2 = v * v;
        let inv_x3 = inv_x2 * v;
        let inv_x4 = inv_x3 * v;
        let d1sq = self.d1 * self.d1;
        let d1 = -inv_x2 * self.d1;
        let d2 = 2.0 * inv_x3 * d1sq - inv_x2 * self.d2;
        let d3 =
            -6.0 * inv_x4 * d1sq * self.d1 + 6.0 * inv_x3 * self.d1 * self.d2 - inv_x2 * self.d3;
        Self { v, d1, d2, d3 }
    }
}

#[inline(always)]
pub fn sin(x: Jet3) -> Jet3 {
    x.sin()
}

#[inline(always)]
pub fn cos(x: Jet3) -> Jet3 {
    x.cos()
}

#[inline(always)]
pub fn exp(x: Jet3) -> Jet3 {
    x.exp()
}

#[inline(always)]
pub fn ln(x: Jet3) -> Jet3 {
    x.ln()
}

#[inline(always)]
pub fn sqrt(x: Jet3) -> Jet3 {
    x.sqrt()
}

#[inline(always)]
pub fn powi(x: Jet3, n: i32) -> Jet3 {
    x.powi(n)
}

impl From<f64> for Jet3 {
    #[inline(always)]
    fn from(value: f64) -> Self {
        Self::constant(value)
    }
}

impl Add for Jet3 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            v: self.v + rhs.v,
            d1: self.d1 + rhs.d1,
            d2: self.d2 + rhs.d2,
            d3: self.d3 + rhs.d3,
        }
    }
}

impl Add<f64> for Jet3 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: f64) -> Self::Output {
        Self {
            v: self.v + rhs,
            ..self
        }
    }
}

impl Add<Jet3> for f64 {
    type Output = Jet3;

    #[inline(always)]
    fn add(self, rhs: Jet3) -> Self::Output {
        rhs + self
    }
}

impl Sub for Jet3 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            v: self.v - rhs.v,
            d1: self.d1 - rhs.d1,
            d2: self.d2 - rhs.d2,
            d3: self.d3 - rhs.d3,
        }
    }
}

impl Sub<f64> for Jet3 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: f64) -> Self::Output {
        Self {
            v: self.v - rhs,
            ..self
        }
    }
}

impl Sub<Jet3> for f64 {
    type Output = Jet3;

    #[inline(always)]
    fn sub(self, rhs: Jet3) -> Self::Output {
        Jet3::constant(self) - rhs
    }
}

impl Mul for Jet3 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            v: self.v * rhs.v,
            d1: self.d1 * rhs.v + self.v * rhs.d1,
            d2: self.d2 * rhs.v + 2.0 * self.d1 * rhs.d1 + self.v * rhs.d2,
            d3: self.d3 * rhs.v + 3.0 * (self.d2 * rhs.d1 + self.d1 * rhs.d2) + self.v * rhs.d3,
        }
    }
}

impl Mul<f64> for Jet3 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            v: self.v * rhs,
            d1: self.d1 * rhs,
            d2: self.d2 * rhs,
            d3: self.d3 * rhs,
        }
    }
}

impl Mul<Jet3> for f64 {
    type Output = Jet3;

    #[inline(always)]
    fn mul(self, rhs: Jet3) -> Self::Output {
        rhs * self
    }
}

impl Div for Jet3 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inv()
    }
}

impl Div<f64> for Jet3 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: f64) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl Div<Jet3> for f64 {
    type Output = Jet3;

    #[inline(always)]
    fn div(self, rhs: Jet3) -> Self::Output {
        Jet3::constant(self) / rhs
    }
}

impl Neg for Jet3 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self {
            v: -self.v,
            d1: -self.d1,
            d2: -self.d2,
            d3: -self.d3,
        }
    }
}
