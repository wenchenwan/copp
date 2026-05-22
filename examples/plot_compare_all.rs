//! Runs all five solvers (TOPP2-RA, COPP2-SOCP, TOPP3-LP, TOPP3-SOCP, COPP3-SOCP)
//! on the same 3-axis Lissajous path and plots speed profiles and trajectories
//! for side-by-side comparison.
//! Output: output/plot_compare_all.png

use std::f64::consts::PI;
use std::fs;
use copp::InterpolationMode;
use copp::path::{Jet3, Path, sin};
use copp::robot::Robot;
use copp::solver::topp2_ra::*;
use copp::solver::copp2_socp::*;
use copp::solver::topp3_lp::*;
use copp::solver::topp3_socp::*;
use copp::solver::copp3_socp::*;
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const DIM: usize = 3;
    let n = 1001;
    let dt = 1e-3;
    let s: Vec<f64> = (0..n).map(|j| j as f64 / (n - 1) as f64).collect();

    // Lissajous path — evaluate up to 3rd order for TOPP3/COPP3
    let path = Path::from_parametric(
        |p: Jet3| {
            vec![
                sin(2.0 * PI * p + 0.0),
                sin(3.0 * PI * p + 0.3),
                sin(5.0 * PI * p + 0.7),
            ]
        },
        0.0,
        1.0,
    )?;
    let derivs = path.evaluate_up_to_3rd(&s)?;

    let vel_max = vec![1.0f64; DIM];
    let vel_min = vec![-1.0f64; DIM];
    let acc_max = vec![1.0f64; DIM];
    let acc_min = vec![-1.0f64; DIM];
    let jerk_max = vec![1.0f64; DIM];
    let jerk_min = vec![-1.0f64; DIM];

    let mut robot = Robot::with_capacity(DIM, n);
    robot.with_s(&s)?;
    robot.with_q(
        &derivs.q.as_view(),
        &derivs.dq.as_ref().unwrap().as_view(),
        &derivs.ddq.as_ref().unwrap().as_view(),
        derivs.dddq.as_ref().map(|m| m.as_view()).as_ref(),
        0,
    )?;
    robot.with_axial_velocity((vel_max.as_slice(), n), (vel_min.as_slice(), n), 0)?;
    robot.with_axial_acceleration((acc_max.as_slice(), n), (acc_min.as_slice(), n), 0)?;
    robot.with_axial_jerk((jerk_max.as_slice(), n), (jerk_min.as_slice(), n), 0)?;

    let idx_s = (0, n - 1);
    let a_bnd = (0.0, 0.0);
    let b_bnd = (0.0, 0.0);
    let clarabel_opts = ClarabelOptionsBuilder::new()
        .allow_almost_solved(true)
        .allow_max_iterations(true)
        .allow_insufficient_progress(true)
        .build()?;

    // 1. TOPP2-RA
    let a_ra = {
        let p = Topp2ProblemBuilder::new(&robot, idx_s, a_bnd).build()?;
        topp2_ra(&p, &ReachSet2OptionsBuilder::new().build()?)?
    };
    let (tf_ra, ts_ra) = s_to_t_topp2(&s, &a_ra, 0.0);
    let st_ra = t_to_s_topp2(&s, &a_ra, &ts_ra, InterpolationMode::UniformTimeGrid(0.0, dt, true));

    // 2. COPP2-SOCP (time-optimal objective)
    let a_copp2 = {
        let obj = [CoppObjective::Time(1.0)];
        let p = Copp2ProblemBuilder::new(&robot, idx_s, a_bnd, &obj).build()?;
        copp2_socp(&p, &clarabel_opts)?
    };
    let (tf_copp2, ts_copp2) = s_to_t_topp2(&s, &a_copp2, 0.0);
    let st_copp2 = t_to_s_topp2(
        &s,
        &a_copp2,
        &ts_copp2,
        InterpolationMode::UniformTimeGrid(0.0, dt, true),
    );

    // Tighten upper bound for 3rd-order solvers using TOPP2-RA result
    robot.constraints.amax_substitute(&a_ra, 0)?;

    // 3. TOPP3-LP (1st SCP iteration)
    let (a_lp, b_lp, num_lp) = {
        let p = Topp3ProblemBuilder::new(&mut robot, 0, &a_ra, a_bnd, b_bnd)
            .build_with_linearization()?;
        topp3_lp(&p, &clarabel_opts)?
    };
    let (tf_lp, ts_lp) = s_to_t_topp3(&s, &a_lp, &b_lp, num_lp, 0.0);
    let st_lp = t_to_s_topp3(
        &s,
        &a_lp,
        &b_lp,
        num_lp,
        &ts_lp,
        InterpolationMode::UniformTimeGrid(0.0, dt, true),
    );

    // 4. TOPP3-SOCP (1st SCP iteration)
    let (a_socp3, b_socp3, num_socp3) = {
        let p = Topp3ProblemBuilder::new(&mut robot, 0, &a_ra, a_bnd, b_bnd)
            .build_with_linearization()?;
        topp3_socp(&p, &clarabel_opts)?
    };
    let (tf_socp3, ts_socp3) = s_to_t_topp3(&s, &a_socp3, &b_socp3, num_socp3, 0.0);
    let st_socp3 = t_to_s_topp3(
        &s,
        &a_socp3,
        &b_socp3,
        num_socp3,
        &ts_socp3,
        InterpolationMode::UniformTimeGrid(0.0, dt, true),
    );

    // 5. COPP3-SOCP (time + thermal energy objective)
    let weights = vec![1.0f64; DIM];
    let (a_copp3, b_copp3, num_copp3) = {
        let obj = [
            CoppObjective::Time(1.0),
            CoppObjective::ThermalEnergy(0.1, &weights),
        ];
        let p = Copp3ProblemBuilder::new(&mut robot, &obj, 0, &a_ra, a_bnd, b_bnd)
            .build_with_linearization()?;
        copp3_socp(&p, &clarabel_opts)?
    };
    let (tf_copp3, ts_copp3) = s_to_t_topp3(&s, &a_copp3, &b_copp3, num_copp3, 0.0);
    let st_copp3 = t_to_s_topp3(
        &s,
        &a_copp3,
        &b_copp3,
        num_copp3,
        &ts_copp3,
        InterpolationMode::UniformTimeGrid(0.0, dt, true),
    );

    println!("t_final (s):");
    println!("  TOPP2-RA  = {tf_ra:.4}");
    println!("  COPP2-SOCP= {tf_copp2:.4}");
    println!("  TOPP3-LP  = {tf_lp:.4}");
    println!("  TOPP3-SOCP= {tf_socp3:.4}");
    println!("  COPP3-SOCP= {tf_copp3:.4}  (time+energy objective)");

    // ---------- Plot ----------
    fs::create_dir_all("output")?;
    let root =
        BitMapBackend::new("output/plot_compare_all.png", (1400, 680)).into_drawing_area();
    root.fill(&WHITE)?;
    let (left, right) = root.split_horizontally(700);

    // Color palette: [TOPP2-RA, COPP2-SOCP, TOPP3-LP, TOPP3-SOCP, COPP3-SOCP]
    let colors = [
        RGBColor(0, 100, 200),
        RGBColor(200, 50, 50),
        RGBColor(0, 160, 80),
        RGBColor(200, 130, 0),
        RGBColor(130, 0, 180),
    ];
    let labels = [
        format!("TOPP2-RA    tf={tf_ra:.3}s"),
        format!("COPP2-SOCP  tf={tf_copp2:.3}s"),
        format!("TOPP3-LP    tf={tf_lp:.3}s"),
        format!("TOPP3-SOCP  tf={tf_socp3:.3}s"),
        format!("COPP3-SOCP  tf={tf_copp3:.3}s (time+energy)"),
    ];

    let speed_profiles: Vec<Vec<f64>> = vec![
        a_ra.iter().map(|&a| a.sqrt()).collect(),
        a_copp2.iter().map(|&a| a.sqrt()).collect(),
        a_lp.iter().map(|&a| a.sqrt()).collect(),
        a_socp3.iter().map(|&a| a.sqrt()).collect(),
        a_copp3.iter().map(|&a| a.sqrt()).collect(),
    ];
    let traj: Vec<&Vec<f64>> = vec![&st_ra, &st_copp2, &st_lp, &st_socp3, &st_copp3];
    let t_finals = [tf_ra, tf_copp2, tf_lp, tf_socp3, tf_copp3];

    let ymax_speed = speed_profiles
        .iter()
        .flat_map(|v| v.iter())
        .cloned()
        .fold(0.0f64, f64::max)
        * 1.12;
    let t_max = t_finals.iter().cloned().fold(0.0f64, f64::max) * 1.02;

    // Left: speed profiles ṡ(s)
    {
        let mut chart = ChartBuilder::on(&left)
            .caption("Speed Profile  ṡ(s) = √a(s)", ("sans-serif", 18))
            .margin(15)
            .x_label_area_size(32)
            .y_label_area_size(55)
            .build_cartesian_2d(0.0f64..1.0f64, 0.0f64..ymax_speed)?;
        chart
            .configure_mesh()
            .x_desc("s (path parameter)")
            .y_desc("ṡ (path speed)")
            .draw()?;

        for (i, (spd, lbl)) in speed_profiles.iter().zip(labels.iter()).enumerate() {
            let c = colors[i];
            chart
                .draw_series(LineSeries::new(
                    s.iter().zip(spd.iter()).map(|(&x, &y)| (x, y)),
                    c.stroke_width(if i < 2 { 2 } else { 1 }),
                ))?
                .label(lbl.as_str())
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 22, y)], c.stroke_width(2))
                });
        }
        chart
            .configure_series_labels()
            .border_style(BLACK)
            .background_style(WHITE.mix(0.88))
            .draw()?;
    }

    // Right: trajectories s(t)
    {
        let mut chart = ChartBuilder::on(&right)
            .caption("Trajectory  s(t)", ("sans-serif", 18))
            .margin(15)
            .x_label_area_size(32)
            .y_label_area_size(45)
            .build_cartesian_2d(0.0f64..t_max, 0.0f64..1.05)?;
        chart
            .configure_mesh()
            .x_desc("t (s)")
            .y_desc("s(t)")
            .draw()?;

        for (i, (st, lbl)) in traj.iter().zip(labels.iter()).enumerate() {
            let c = colors[i];
            chart
                .draw_series(LineSeries::new(
                    (0..st.len()).map(|j| (j as f64 * dt, st[j])),
                    c.stroke_width(if i < 2 { 2 } else { 1 }),
                ))?
                .label(lbl.as_str())
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 22, y)], c.stroke_width(2))
                });
        }
        chart
            .configure_series_labels()
            .border_style(BLACK)
            .background_style(WHITE.mix(0.88))
            .draw()?;
    }

    root.present()?;
    println!("Saved: output/plot_compare_all.png");
    Ok(())
}
