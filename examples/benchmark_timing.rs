//! 各求解器计算耗时对比基准测试。
//!
//! # 测量范围
//! 对每个求解器统计：`problem build`（含 `build_with_linearization`）+ 优化器调用。
//!
//! **不计入**（所有方法公共，不影响对比）：
//! - 路径求值 `path.evaluate_up_to_3rd`
//! - robot 约束摄入（`with_s / with_q / with_axial_*`）
//! - `amax_substitute`（3 阶方法共享的预处理步骤）
//!
//! # 方法列表
//! | 方法 | 描述 |
//! |------|------|
//! | TOPP2-RA | 后向 DP 可达集 + 前向贪心，O(n·m) |
//! | COPP2-SOCP | 二阶凸目标（时间最优），Clarabel SOCP |
//! | TOPP3-LP  iter1/2 | 三阶 LP，第 1/2 次 SCP 迭代 |
//! | TOPP3-SOCP iter1/2 | 三阶 SOCP，第 1/2 次 SCP 迭代 |
//! | COPP3-SOCP iter1/2 | 三阶凸目标（时间+热能），第 1/2 次 SCP 迭代 |
//!
//! # 运行方式
//! ```bash
//! cargo run --example benchmark_timing --release
//! ```
//!
//! # 输出
//! - 控制台：格式化耗时对比表格
//! - 文件：`output/benchmark_timing.png`（水平条形图）

use std::f64::consts::PI;
use std::fs;
use std::time::Instant;

use copp::path::{Jet3, Path, sin};
use copp::robot::Robot;
use copp::solver::copp2_socp::*;
use copp::solver::copp3_socp::*;
use copp::solver::topp2_ra::*;
use copp::solver::topp3_lp::*;
use copp::solver::topp3_socp::*;
use plotters::prelude::*;

/// 单条基准记录。
struct BenchEntry {
    label: &'static str,
    /// 求解耗时（毫秒）。
    solve_ms: f64,
    /// 最终运动时长（秒）。
    t_final: f64,
    /// true 表示汇总行（两次迭代之和），不参与单独条形图。
    is_summary: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const DIM: usize = 3;
    const N: usize = 1001;

    println!("建立路径与约束（共用预处理，不计时）…");

    // ── 公共路径与约束（不计时）──────────────────────────────────────────────
    let path = Path::from_parametric(
        |p: Jet3| {
            vec![
                sin(2.0 * PI * p),
                sin(3.0 * PI * p + 0.3),
                sin(5.0 * PI * p + 0.7),
            ]
        },
        0.0,
        1.0,
    )?;
    let s: Vec<f64> = (0..N).map(|j| j as f64 / (N - 1) as f64).collect();
    let derivs = path.evaluate_up_to_3rd(&s)?;

    let mut robot = Robot::with_capacity(DIM, N);
    robot.with_s(&s)?;
    robot.with_q(
        &derivs.q.as_view(),
        &derivs.dq.as_ref().unwrap().as_view(),
        &derivs.ddq.as_ref().unwrap().as_view(),
        derivs.dddq.as_ref().map(|m| m.as_view()).as_ref(),
        0,
    )?;
    let lim = vec![1.0f64; DIM];
    let neg = vec![-1.0f64; DIM];
    robot.with_axial_velocity((lim.as_slice(), N), (neg.as_slice(), N), 0)?;
    robot.with_axial_acceleration((lim.as_slice(), N), (neg.as_slice(), N), 0)?;
    robot.with_axial_jerk((lim.as_slice(), N), (neg.as_slice(), N), 0)?;

    let a_bnd = (0.0f64, 0.0f64);
    let b_bnd = (0.0f64, 0.0f64);
    let ra_opts = ReachSet2OptionsBuilder::new().build()?;
    let clarabel_opts = ClarabelOptionsBuilder::new()
        .allow_almost_solved(true)
        .allow_max_iterations(true)
        .allow_insufficient_progress(true)
        .build()?;
    let weights = vec![1.0f64; DIM];

    let mut bench: Vec<BenchEntry> = Vec::new();

    println!("开始各求解器计时…\n");

    // ── 1. TOPP2-RA ──────────────────────────────────────────────────────────
    // 注意：使用块将 `p` 的生命周期限制在内部，以便后续 &mut robot 使用
    let a_ra = {
        let t0 = Instant::now();
        let p = Topp2ProblemBuilder::new(&robot, (0, N - 1), a_bnd).build()?;
        let a = topp2_ra(&p, &ra_opts)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let (tf, _) = s_to_t_topp2(&s, &a, 0.0);
        println!("  TOPP2-RA         {:8.2} ms  tf={:.4} s", ms, tf);
        bench.push(BenchEntry { label: "TOPP2-RA", solve_ms: ms, t_final: tf, is_summary: false });
        a
    };
    let tf_ra = bench[0].t_final;

    // ── 2. COPP2-SOCP ────────────────────────────────────────────────────────
    {
        let obj = [CoppObjective::Time(1.0)];
        let t0 = Instant::now();
        let p = Copp2ProblemBuilder::new(&robot, (0, N - 1), a_bnd, &obj).build()?;
        let a = copp2_socp(&p, &clarabel_opts)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let (tf, _) = s_to_t_topp2(&s, &a, 0.0);
        println!("  COPP2-SOCP       {:8.2} ms  tf={:.4} s", ms, tf);
        bench.push(BenchEntry {
            label: "COPP2-SOCP",
            solve_ms: ms,
            t_final: tf,
            is_summary: false,
        });
    }

    // 收紧一阶上界（不计时），为 3 阶方法提供更紧参考域
    robot.constraints.amax_substitute(&a_ra, 0)?;

    // ── 3. TOPP3-LP ──────────────────────────────────────────────────────────
    // 使用块确保 `p`（借用 &mut robot）在离开块时释放，再进行下一次 &mut robot 借用
    let (a_lp1, ms_lp1, tf_lp1) = {
        let t0 = Instant::now();
        let p = Topp3ProblemBuilder::new(&mut robot, 0, &a_ra, a_bnd, b_bnd)
            .build_with_linearization()?;
        let (a, b, num) = topp3_lp(&p, &clarabel_opts)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let (tf, _) = s_to_t_topp3(&s, &a, &b, num, 0.0);
        (a, ms, tf)
    };
    println!("  TOPP3-LP  iter1  {:8.2} ms  tf={:.4} s", ms_lp1, tf_lp1);
    bench.push(BenchEntry {
        label: "TOPP3-LP  iter1",
        solve_ms: ms_lp1,
        t_final: tf_lp1,
        is_summary: false,
    });

    let (ms_lp2, tf_lp2) = {
        let t0 = Instant::now();
        let p = Topp3ProblemBuilder::new(&mut robot, 0, &a_lp1, a_bnd, b_bnd)
            .build_with_linearization()?;
        let (a, b, num) = topp3_lp(&p, &clarabel_opts)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let (tf, _) = s_to_t_topp3(&s, &a, &b, num, 0.0);
        (ms, tf)
    };
    println!("  TOPP3-LP  iter2  {:8.2} ms  tf={:.4} s", ms_lp2, tf_lp2);
    bench.push(BenchEntry {
        label: "TOPP3-LP  iter2",
        solve_ms: ms_lp2,
        t_final: tf_lp2,
        is_summary: false,
    });
    bench.push(BenchEntry {
        label: "TOPP3-LP  total",
        solve_ms: ms_lp1 + ms_lp2,
        t_final: tf_lp2,
        is_summary: true,
    });

    // ── 4. TOPP3-SOCP ────────────────────────────────────────────────────────
    let (a_s1, ms_s1, tf_s1) = {
        let t0 = Instant::now();
        let p = Topp3ProblemBuilder::new(&mut robot, 0, &a_ra, a_bnd, b_bnd)
            .build_with_linearization()?;
        let (a, b, num) = topp3_socp(&p, &clarabel_opts)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let (tf, _) = s_to_t_topp3(&s, &a, &b, num, 0.0);
        (a, ms, tf)
    };
    println!("  TOPP3-SOCP iter1 {:8.2} ms  tf={:.4} s", ms_s1, tf_s1);
    bench.push(BenchEntry {
        label: "TOPP3-SOCP iter1",
        solve_ms: ms_s1,
        t_final: tf_s1,
        is_summary: false,
    });

    let (ms_s2, tf_s2) = {
        let t0 = Instant::now();
        let p = Topp3ProblemBuilder::new(&mut robot, 0, &a_s1, a_bnd, b_bnd)
            .build_with_linearization()?;
        let (a, b, num) = topp3_socp(&p, &clarabel_opts)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let (tf, _) = s_to_t_topp3(&s, &a, &b, num, 0.0);
        (ms, tf)
    };
    println!("  TOPP3-SOCP iter2 {:8.2} ms  tf={:.4} s", ms_s2, tf_s2);
    bench.push(BenchEntry {
        label: "TOPP3-SOCP iter2",
        solve_ms: ms_s2,
        t_final: tf_s2,
        is_summary: false,
    });
    bench.push(BenchEntry {
        label: "TOPP3-SOCP total",
        solve_ms: ms_s1 + ms_s2,
        t_final: tf_s2,
        is_summary: true,
    });

    // ── 5. COPP3-SOCP（时间 + 热能目标）─────────────────────────────────────
    let obj3 = [
        CoppObjective::Time(1.0),
        CoppObjective::ThermalEnergy(0.1, &weights),
    ];

    let (a_c1, ms_c1, tf_c1) = {
        let t0 = Instant::now();
        let p = Copp3ProblemBuilder::new(&mut robot, &obj3, 0, &a_ra, a_bnd, b_bnd)
            .build_with_linearization()?;
        let (a, b, num) = copp3_socp(&p, &clarabel_opts)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let (tf, _) = s_to_t_topp3(&s, &a, &b, num, 0.0);
        (a, ms, tf)
    };
    println!("  COPP3-SOCP iter1 {:8.2} ms  tf={:.4} s", ms_c1, tf_c1);
    bench.push(BenchEntry {
        label: "COPP3-SOCP iter1",
        solve_ms: ms_c1,
        t_final: tf_c1,
        is_summary: false,
    });

    let (ms_c2, tf_c2) = {
        let t0 = Instant::now();
        let p = Copp3ProblemBuilder::new(&mut robot, &obj3, 0, &a_c1, a_bnd, b_bnd)
            .build_with_linearization()?;
        let (a, b, num) = copp3_socp(&p, &clarabel_opts)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let (tf, _) = s_to_t_topp3(&s, &a, &b, num, 0.0);
        (ms, tf)
    };
    println!("  COPP3-SOCP iter2 {:8.2} ms  tf={:.4} s", ms_c2, tf_c2);
    bench.push(BenchEntry {
        label: "COPP3-SOCP iter2",
        solve_ms: ms_c2,
        t_final: tf_c2,
        is_summary: false,
    });
    bench.push(BenchEntry {
        label: "COPP3-SOCP total",
        solve_ms: ms_c1 + ms_c2,
        t_final: tf_c2,
        is_summary: true,
    });

    // ── 控制台格式化表格 ──────────────────────────────────────────────────────
    let sep = "─".repeat(70);
    println!("\n┌{sep}┐");
    println!(
        "│  COPP 求解器耗时对比   n={N}, dim={DIM}, Lissajous 路径, --release{:>8}│",
        ""
    );
    println!("├{sep}┤");
    println!(
        "│  {:<22} {:>10}  {:>12}  {:>10}  │",
        "求解器", "耗时(ms)", "t_final(s)", "t/t_ra"
    );
    println!("├{sep}┤");
    for e in &bench {
        let mark = if e.is_summary { "→" } else { " " };
        println!(
            "│{mark} {:<22} {:>10.2}  {:>12.4}  {:>10.4}  │",
            e.label,
            e.solve_ms,
            e.t_final,
            e.t_final / tf_ra
        );
    }
    println!("└{sep}┘");

    // ── 条形图（仅显示迭代行，不含汇总行）──────────────────────────────────
    let bar_entries: Vec<&BenchEntry> = bench.iter().filter(|e| !e.is_summary).collect();
    let nb = bar_entries.len();

    let max_ms = bar_entries
        .iter()
        .map(|e| e.solve_ms)
        .fold(0.0f64, f64::max);
    let x_max = max_ms * 1.25; // 留出标注文字空间

    fs::create_dir_all("output")?;
    let root =
        BitMapBackend::new("output/benchmark_timing.png", (1000, 540)).into_drawing_area();
    root.fill(&WHITE)?;

    // 每个求解器家族用一组颜色区分
    let bar_colors: &[RGBColor] = &[
        RGBColor(30, 110, 210),  // TOPP2-RA
        RGBColor(210, 50, 50),   // COPP2-SOCP
        RGBColor(0, 155, 75),    // TOPP3-LP  iter1
        RGBColor(60, 195, 115),  // TOPP3-LP  iter2
        RGBColor(210, 125, 0),   // TOPP3-SOCP iter1
        RGBColor(235, 165, 35),  // TOPP3-SOCP iter2
        RGBColor(125, 0, 175),   // COPP3-SOCP iter1
        RGBColor(165, 45, 215),  // COPP3-SOCP iter2
    ];

    // y 轴标签（逆序，使第 0 条在顶部）
    let y_labels: Vec<String> = bar_entries.iter().rev().map(|e| e.label.to_string()).collect();

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("COPP 求解器耗时对比  (n={N}, dim={DIM}, --release)"),
            ("sans-serif", 18),
        )
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(180)
        .build_cartesian_2d(0.0f64..x_max, -0.6f64..(nb as f64 - 0.4))?;

    chart
        .configure_mesh()
        .disable_y_mesh()
        .x_desc("求解耗时 (ms)")
        .y_labels(nb)
        .y_label_formatter(&|y: &f64| {
            // y 的整数部分对应从底部数的条形索引（底部 = 最后一个求解器）
            let idx = y.round();
            if idx < 0.0 || idx as usize >= y_labels.len() {
                String::new()
            } else {
                y_labels[idx as usize].clone()
            }
        })
        .draw()?;

    for (i, e) in bar_entries.iter().enumerate() {
        // 逆序排列：第 0 个求解器画在最顶部（y_center = nb-1）
        let y_center = (nb - 1 - i) as f64;
        let color = bar_colors[i % bar_colors.len()];

        // 绘制填充矩形
        chart.draw_series(std::iter::once(Rectangle::new(
            [(0.0, y_center - 0.38), (e.solve_ms, y_center + 0.38)],
            color.filled(),
        )))?;

        // 数值标注（显示在条形右侧）
        let label_x = e.solve_ms + x_max * 0.012;
        let label_text = format!("{:.1} ms", e.solve_ms);
        chart.draw_series(std::iter::once(Text::new(
            label_text,
            (label_x, y_center - 0.18),
            ("sans-serif", 12).into_font(),
        )))?;
    }

    root.present()?;
    println!("\nSaved: output/benchmark_timing.png");
    Ok(())
}
