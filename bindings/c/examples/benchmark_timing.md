# benchmark_timing — 求解器耗时对比

> 源文件：[`benchmark_timing.c`](benchmark_timing.c)  
> CMake 目标：`benchmark_timing`

本示例在同一条 Lissajous 测试路径上依次运行所有开源 COPP/TOPP 求解器，统计各阶段的 wall-clock 耗时，并打印对比表格。

---

## 目录

- [测试场景](#测试场景)
- [计时范围](#计时范围)
- [求解器列表与算法简介](#求解器列表与算法简介)
- [SCP 迭代结构](#scp-迭代结构)
- [构建与运行](#构建与运行)
- [输出格式说明](#输出格式说明)
- [典型结果参考](#典型结果参考)
- [计时器实现](#计时器实现)
- [常见问题](#常见问题)

---

## 测试场景

所有求解器共用同一条解析 3 轴 Lissajous 路径，与其他 C 示例一致：

$$
q_0(s) = \sin(2\pi s), \quad
q_1(s) = \sin(3\pi s + 0.3), \quad
q_2(s) = \sin(5\pi s + 0.7)
$$

参数设置：

| 参数 | 值 |
|------|----|
| 路径区间 | $s \in [0,\ 1]$ |
| 离散站点数 $N$ | 1001 |
| 机器人自由度 | 3 |
| 轴速度限 | $[-1,\ 1]$ |
| 轴加速度限 | $[-1,\ 1]$ |
| 轴加加速度限（三阶方法） | $[-1,\ 1]$ |
| 起止边界 | $a(0) = a(1) = 0,\quad b(0) = b(1) = 0$ |

---

## 计时范围

```
┌─────────────────────────────────────────────────┐
│  路径求值 + robot 约束摄入（所有方法共用，不计时）    │
├─────────────────────────────────────────────────┤
│  amax_substitute（三阶方法共用预处理，不计时）        │
├─────────────────────────────────────────────────┤
│  ► 计时开始                                       │
│     problem 描述符初始化                           │
│     + 约束线性化（三阶方法内部执行）                 │
│     + 求解器调用                                   │
│  ◄ 计时结束                                       │
└─────────────────────────────────────────────────┘
```

**不计入计时**的操作对所有方法完全相同，不影响横向对比：

- `copp_path_from_evaluator_*` — 路径句柄创建
- `copp_robot_create` / `copp_robot_sample_path_*` — 路径导数采样
- `copp_add_axial_*_limits` — 约束摄入
- `copp_robot_amax_substitute` — 一阶上界收紧（三阶方法共用）
- `copp_s_to_t_*` / `copp_t_to_s_*` — 结果后处理

---

## 求解器列表与算法简介

### TOPP2-RA

**时间最优二阶路径参数化，后向可达集 + 前向贪心**

路径参数化变量：

$$
a(s) = \dot{s}^2, \quad b(s) = \ddot{s}
$$

速度、加速度约束经链式法则转化为 $a$ 的一阶和二阶不等式：

$$
a(s) \le a_{\max}(s), \qquad
\alpha_k\,a_k + \beta_k\,b_{k-1} \le \gamma_k
$$

算法步骤：
1. **后向传播**：从终点逐站计算可达上界 $a_{\max}^{(B)}[k]$，每步调用一个二维 Seidel LP（期望 $O(m)$，$m$ 为该站约束数）。
2. **前向贪心**：从起点沿可达上界取最大可行值，得到时间最优轮廓 $a[k]$。

整体复杂度：$O(N \cdot m)$，速度极快，通常在亚毫秒级完成。

---

### COPP2-SOCP

**凸目标二阶路径参数化，Clarabel SOCP**

与 TOPP2-RA 使用相同变量 $a(s)$，但将时间目标

$$
\min\; w \int_0^1 \frac{ds}{\sqrt{a(s)}}
$$

通过辅助变量 $\xi_k \ge 1/\sqrt{a_k}$ 转化为二阶锥约束（SOCP），交由 Clarabel 求解。

---

### TOPP3-LP / TOPP3-SOCP

**时间最优三阶路径参数化，加加速度约束，Clarabel LP / SOCP**

三阶方法引入额外变量：

$$
c(s) = \frac{\dddot{s}}{\dot{s}}
$$

加加速度约束为非凸的：

$$
\sqrt{a}\bigl(g_a\,a + g_b\,b + g_c\,c + g_d\bigr) \le g_{\max}
$$

通过**序列凸规划（SCP）**在参考轮廓 $a_{\mathrm{lin}}$ 处线性化：

$$
\underbrace{\Bigl(g_a + \frac{g_{\max}}{2\,a_{\mathrm{lin}}^{3/2}}\Bigr)}_{h_a} a
+ g_b\,b + g_c\,c
\le
\underbrace{\frac{3}{2}\frac{g_{\max}}{\sqrt{a_{\mathrm{lin}}}} - g_d}_{h_{\max}}
$$

线性化后：

- **TOPP3-LP**：目标和约束均为线性，使用 Clarabel LP 后端（无辅助锥变量）。
- **TOPP3-SOCP**：时间目标通过辅助变量 $\xi_k \ge 1/\sqrt{a_k}$ 转化为 SOCP，求解更精确但规模更大。

决策变量布局：

$$
x = [\,a_0,\,a_1,\,\ldots,\,a_N,\;b_0,\,b_1,\,\ldots,\,b_N\,]^{\top}
$$

---

### COPP3-SOCP

**凸目标三阶路径参数化，Clarabel SOCP**

在 TOPP3-SOCP 基础上支持复合凸目标，本示例使用：

$$
\min\quad w_t \int_0^1 \frac{ds}{\sqrt{a}}
+ w_e \int_0^1 \frac{\displaystyle\sum_i (\tau_i\,\nu_i)^2}{\sqrt{a}}\,ds
$$

其中 $\tau_i$ 为广义力（默认点动力学 $\tau = \ddot{q}$），$\nu_i$ 为归一化权重。

本示例参数：$w_t = 1.0$，$w_e = 0.1$，$\nu_i = 1$。

---

## SCP 迭代结构

三阶方法（TOPP3-LP / TOPP3-SOCP / COPP3-SOCP）均需迭代：

```
iter 1：以 TOPP2-RA 结果 a_ra 作为线性化点
         → 得到轮廓 a_1（三阶可行但保守）

iter 2：以 a_1 作为新线性化点
         → 得到轮廓 a_2（更接近三阶最优）
```

每次迭代独立计时，表格中还给出 `total`（iter 1 + iter 2 的合计耗时），用 `->` 标注。实际工程中通常 2 次迭代已足够收敛。

---

## 构建与运行

### 第一步：编译 Rust 动态库

在仓库根目录：

```sh
cargo build --release
```

### 第二步：CMake 配置与编译

```sh
# 从仓库根目录
cmake -S bindings/c -B bindings/c/build
cmake --build bindings/c/build --config Release --target benchmark_timing
```

仅编译本示例，跳过测试：

```sh
cmake -S bindings/c -B bindings/c/build -DCOPP_BUILD_TESTS=OFF
cmake --build bindings/c/build --config Release --target benchmark_timing
```

### 第三步：运行

Windows（动态链接，`copp.dll` 已由 CMake 自动复制到可执行文件目录）：

```bat
bindings\c\build\Release\benchmark_timing.exe
```

Linux / macOS：

```sh
./bindings/c/build/benchmark_timing
```

**建议**：使用 Release 构建运行，Debug 构建中 Clarabel 内部迭代未优化，耗时参考价值有限。

---

## 输出格式说明

程序输出分两部分。

### 进度行

每个求解器完成后立即打印一行：

```
  TOPP2-RA           0.31 ms  tf=15.8596 s
  COPP2-SOCP (Time)  246.57 ms  tf=15.8596 s
  ...
```

### 汇总表格

```
+---------------------+----------+------------+---------+
|  COPP 求解器耗时对比   n=1001  dim=3  Lissajous       |
+---------------------+----------+------------+---------+
|  求解器              |  耗时(ms)|  t_final(s)|  t/t_ra |
+---------------------+----------+------------+---------+
|  TOPP2-RA           |     0.31 |    15.8596 |  1.0000 |
|  COPP2-SOCP (Time)  |   246.57 |    15.8596 |  1.0000 |
|  TOPP3-LP  iter 1   |   113.71 |    17.9943 |  1.1346 |
|  TOPP3-LP  iter 2   |   138.80 |    17.6291 |  1.1116 |
|-> TOPP3-LP  total   |   252.51 |    17.6291 |  1.1116 |
|  TOPP3-SOCP iter 1  |   263.05 |    17.9943 |  1.1346 |
|  TOPP3-SOCP iter 2  |   207.91 |    17.6291 |  1.1116 |
|-> TOPP3-SOCP total  |   470.97 |    17.6291 |  1.1116 |
|  COPP3-SOCP iter 1  |   308.71 |    17.9943 |  1.1346 |
|  COPP3-SOCP iter 2  |   253.31 |    17.6291 |  1.1116 |
|-> COPP3-SOCP total  |   562.01 |    17.6291 |  1.1116 |
+---------------------+----------+------------+---------+
  -> 行表示两次 SCP 迭代的合计耗时
```

列含义：

| 列 | 说明 |
|----|------|
| `耗时(ms)` | wall-clock 求解时间（毫秒） |
| `t_final(s)` | 求解器给出的路径总遍历时间（秒） |
| `t/t_ra` | 轨迹时长与 TOPP2-RA 结果之比；值越接近 1.0 表示越接近二阶时间最优 |

**注意**：三阶方法的 `t_final` 通常大于 TOPP2-RA（因为加加速度约束更严），`t/t_ra > 1` 是正常现象，不代表求解质量差。

---

## 典型结果参考

以下数据在 Windows 11、Ryzen 9、`--release` 编译、动态链接下测得，仅供参考：

| 求解器 | 耗时(ms) | t_final(s) | t/t_ra |
|--------|----------|------------|--------|
| TOPP2-RA | 0.3 | 15.86 | 1.000 |
| COPP2-SOCP | 247 | 15.86 | 1.000 |
| TOPP3-LP total | 253 | 17.63 | 1.112 |
| TOPP3-SOCP total | 471 | 17.63 | 1.112 |
| COPP3-SOCP total | 562 | 17.63 | 1.112 |

主要观察：

- **TOPP2-RA** 比所有 SOCP/LP 方法快约 3 个数量级，适合实时或在线场景。
- **COPP2-SOCP** 与 TOPP2-RA 轨迹质量相同（`t/t_ra ≈ 1`），但通过 Clarabel 求解，可方便地替换为凸目标。
- **TOPP3-LP** 两次迭代合计耗时约与 COPP2-SOCP 相当，但提供三阶约束满足（加加速度连续）。
- **TOPP3-SOCP** 因引入辅助锥变量，规模大于 TOPP3-LP，耗时约为其 2 倍。
- **COPP3-SOCP** 在 TOPP3-SOCP 基础上增加热能目标项，额外开销约 20%。

---

## 计时器实现

代码中的 `bench_now_ms()` 函数根据平台自动选择高精度计时器：

```c
#ifdef _WIN32
    /* Windows：QueryPerformanceCounter，分辨率约 100 ns */
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    /* POSIX：clock_gettime(CLOCK_MONOTONIC)，分辨率约 1 ns */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
#endif
```

两者均为单调时钟，不受系统时间调整影响，适合短时间段的性能测量。

---

## 常见问题

### 三阶方法返回 `COPP_STATUS_CLARABEL_*` 错误

适当放宽求解容限：

```c
opts3.allow_almost_solved         = true;
opts3.allow_max_iterations        = true;
opts3.allow_insufficient_progress = true;
```

本示例已默认开启以上选项。若仍失败，可增大 `a_linearization_floor`（当前为 `1e-10`）或检查约束可行性。

### TOPP2-RA 耗时为零或极小

TOPP2-RA 的 Seidel LP 为纯 CPU 计算，1001 站点下通常不足 1 ms。若需更精确的微秒级测量，可多次重复调用后取均值（热身 + 多次采样）。

### Debug 构建耗时异常偏高

Clarabel 内部大量使用矩阵运算，Debug 模式未开启优化，耗时可能是 Release 的 10–50 倍。请始终用 Release 构建进行性能对比。
