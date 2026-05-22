//! Runs all six solvers on a 3-axis Lissajous path and saves results to
//! output/data/ as CSV files for downstream Python plotting.
//!
//! Run with:  cargo run --example save_all --release
//!
//! Files produced (output/data/):
//!   topp2_ra.csv / topp2_ra_traj.csv
//!   copp2_socp.csv / copp2_socp_traj.csv
//!   reach_set2.csv
//!   topp3_lp_iter{1,2}.csv / topp3_lp_iter{1,2}_traj.csv
//!   topp3_socp_iter{1,2}.csv / topp3_socp_iter{1,2}_traj.csv
//!   copp3_socp_iter{1,2}.csv / copp3_socp_iter{1,2}_traj.csv

use std::f64::consts::PI;
use std::fs;
use std::io::{BufWriter, Write};
use copp::InterpolationMode;
use copp::path::{Jet3, Path, sin};
use copp::robot::Robot;
use copp::solver::topp2_ra::*;
use copp::solver::copp2_socp::*;
use copp::solver::reach_set2::{reach_set2_backward, reach_set2_bidirectional};
use copp::solver::topp3_lp::*;
use copp::solver::topp3_socp::*;
use copp::solver::copp3_socp::*;

// Write column-aligned CSV: one row per index, one column per slice.
fn save_csv(path: &str, headers: &[&str], cols: &[&[f64]]) -> std::io::Result<()> {
    let n = cols[0].len();
    let f = fs::File::create(path)?;
    let mut w = BufWriter::new(f);
    writeln!(w, "{}", headers.join(","))?;
    for i in 0..n {
        let row: Vec<String> = cols.iter().map(|c| format!("{:.10e}", c[i])).collect();
        writeln!(w, "{}", row.join(","))?;
    }
    Ok(())
}

// Write uniform-time trajectory (t_k = k*dt, s_t[k]).
fn save_traj(path: &str, dt: f64, s_t: &[f64]) -> std::io::Result<()> {
    let f = fs::File::create(path)?;
    let mut w = BufWriter::new(f);
    writeln!(w, "t,s_t")?;
    for (i, &sv) in s_t.iter().enumerate() {
        writeln!(w, "{:.10e},{:.10e}", i as f64 * dt, sv)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const DIM: usize = 3;
    let n = 1001;
    let dt = 1e-3;
    let s: Vec<f64> = (0..n).map(|j| j as f64 / (n - 1) as f64).collect();

    // Lissajous path — need 3rd derivatives for TOPP3/COPP3
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

    let ones = vec![1.0f64; DIM];
    let neg_ones = vec![-1.0f64; DIM];

    let mut robot = Robot::with_capacity(DIM, n);
    robot.with_s(&s)?;
    robot.with_q(
        &derivs.q.as_view(),
        &derivs.dq.as_ref().unwrap().as_view(),
        &derivs.ddq.as_ref().unwrap().as_view(),
        derivs.dddq.as_ref().map(|m| m.as_view()).as_ref(),
        0,
    )?;
    robot.with_axial_velocity((ones.as_slice(), n), (neg_ones.as_slice(), n), 0)?;
    robot.with_axial_acceleration((ones.as_slice(), n), (neg_ones.as_slice(), n), 0)?;
    robot.with_axial_jerk((ones.as_slice(), n), (neg_ones.as_slice(), n), 0)?;

    fs::create_dir_all("output/data")?;

    // Save path geometry (q, q', q'') for joint-space trajectory reconstruction
    {
        let dq  = derivs.dq.as_ref().unwrap();
        let ddq = derivs.ddq.as_ref().unwrap();
        let dim = derivs.q.nrows();
        let mut headers = vec!["s".to_string()];
        for i in 0..dim { headers.push(format!("q{i}")); }
        for i in 0..dim { headers.push(format!("dq{i}")); }
        for i in 0..dim { headers.push(format!("ddq{i}")); }
        let f = fs::File::create("output/data/path_derivs.csv")?;
        let mut w = BufWriter::new(f);
        writeln!(w, "{}", headers.join(","))?;
        for j in 0..n {
            let mut row = vec![format!("{:.10e}", s[j])];
            for i in 0..dim { row.push(format!("{:.10e}", derivs.q.column(j)[i])); }
            for i in 0..dim { row.push(format!("{:.10e}", dq.column(j)[i])); }
            for i in 0..dim { row.push(format!("{:.10e}", ddq.column(j)[i])); }
            writeln!(w, "{}", row.join(","))?;
        }
        println!("path_derivs: saved  →  output/data/path_derivs.csv");
    }

    let idx_s = (0, n - 1);
    let a_bnd = (0.0, 0.0);
    let b_bnd = (0.0, 0.0);
    let clarabel = ClarabelOptionsBuilder::new()
        .allow_almost_solved(true)
        .allow_max_iterations(true)
        .allow_insufficient_progress(true)
        .build()?;

    // Build TOPP2 problem once; reused for RA and reach-set operations.
    let problem2 = Topp2ProblemBuilder::new(&robot, idx_s, a_bnd).build()?;
    let opts_ra = ReachSet2OptionsBuilder::new().build()?;

    // ------------------------------------------------------------------
    // 1. TOPP2-RA
    // ------------------------------------------------------------------
    let a_ra = topp2_ra(&problem2, &opts_ra)?;
    let (tf_ra, t_s_ra) = s_to_t_topp2(&s, &a_ra, 0.0);
    let st_ra = t_to_s_topp2(
        &s,
        &a_ra,
        &t_s_ra,
        InterpolationMode::UniformTimeGrid(0.0, dt, true),
    );
    save_csv(
        "output/data/topp2_ra.csv",
        &["s", "a", "t_s"],
        &[s.as_slice(), &a_ra, &t_s_ra],
    )?;
    save_traj("output/data/topp2_ra_traj.csv", dt, &st_ra)?;
    println!("TOPP2-RA:   t_final = {tf_ra:.4} s");

    // ------------------------------------------------------------------
    // 2. COPP2-SOCP (time-optimal objective)
    // ------------------------------------------------------------------
    let a_copp2 = {
        let obj = [CoppObjective::Time(1.0)];
        let p = Copp2ProblemBuilder::new(&robot, idx_s, a_bnd, &obj).build()?;
        copp2_socp(&p, &clarabel)?
    };
    let (tf_copp2, t_s_copp2) = s_to_t_topp2(&s, &a_copp2, 0.0);
    let st_copp2 = t_to_s_topp2(
        &s,
        &a_copp2,
        &t_s_copp2,
        InterpolationMode::UniformTimeGrid(0.0, dt, true),
    );
    save_csv(
        "output/data/copp2_socp.csv",
        &["s", "a", "t_s"],
        &[s.as_slice(), &a_copp2, &t_s_copp2],
    )?;
    save_traj("output/data/copp2_socp_traj.csv", dt, &st_copp2)?;
    println!("COPP2-SOCP: t_final = {tf_copp2:.4} s");

    // ------------------------------------------------------------------
    // 3. reach_set2 (backward + bidirectional)
    // ------------------------------------------------------------------
    {
        let back = reach_set2_backward(&problem2, &opts_ra)?;
        let bidir = reach_set2_bidirectional(&problem2, &opts_ra)?;
        save_csv(
            "output/data/reach_set2.csv",
            &[
                "s",
                "a_max_back",
                "a_min_back",
                "a_max_bidir",
                "a_min_bidir",
                "a_ra",
                "t_s_ra",
            ],
            &[
                s.as_slice(),
                &back.a_max,
                &back.a_min,
                &bidir.a_max,
                &bidir.a_min,
                &a_ra,
                &t_s_ra,
            ],
        )?;
        println!("reach_set2: saved");
    }

    // Tighten velocity-squared upper bound for all 3rd-order solvers.
    robot.constraints.amax_substitute(&a_ra, 0)?;

    // Helper: save a 3rd-order profile + traj pair.
    let save3 = |tag: &str,
                 a: &[f64],
                 b: &[f64],
                 t_s: &[f64],
                 s_t: &[f64]|
     -> std::io::Result<()> {
        save_csv(
            &format!("output/data/{tag}.csv"),
            &["s", "a", "b", "t_s"],
            &[s.as_slice(), a, b, t_s],
        )?;
        save_traj(&format!("output/data/{tag}_traj.csv"), dt, s_t)
    };

    // ------------------------------------------------------------------
    // 4. TOPP3-LP  (SCP iterations 1 and 2)
    // ------------------------------------------------------------------
    let (a_lp1, b_lp1, num_lp1) = {
        let p = Topp3ProblemBuilder::new(&mut robot, 0, &a_ra, a_bnd, b_bnd)
            .build_with_linearization()?;
        topp3_lp(&p, &clarabel)?
    };
    let (tf_lp1, t_s_lp1) = s_to_t_topp3(&s, &a_lp1, &b_lp1, num_lp1, 0.0);
    let st_lp1 = t_to_s_topp3(
        &s,
        &a_lp1,
        &b_lp1,
        num_lp1,
        &t_s_lp1,
        InterpolationMode::UniformTimeGrid(0.0, dt, true),
    );
    save3("topp3_lp_iter1", &a_lp1, &b_lp1, &t_s_lp1, &st_lp1)?;

    let (a_lp2, b_lp2, num_lp2) = {
        let p = Topp3ProblemBuilder::new(&mut robot, 0, &a_lp1, a_bnd, b_bnd)
            .build_with_linearization()?;
        topp3_lp(&p, &clarabel)?
    };
    let (tf_lp2, t_s_lp2) = s_to_t_topp3(&s, &a_lp2, &b_lp2, num_lp2, 0.0);
    let st_lp2 = t_to_s_topp3(
        &s,
        &a_lp2,
        &b_lp2,
        num_lp2,
        &t_s_lp2,
        InterpolationMode::UniformTimeGrid(0.0, dt, true),
    );
    save3("topp3_lp_iter2", &a_lp2, &b_lp2, &t_s_lp2, &st_lp2)?;
    println!("TOPP3-LP:   iter1 = {tf_lp1:.4} s,  iter2 = {tf_lp2:.4} s");

    // ------------------------------------------------------------------
    // 5. TOPP3-SOCP  (SCP iterations 1 and 2)
    // ------------------------------------------------------------------
    let (a_sq1, b_sq1, num_sq1) = {
        let p = Topp3ProblemBuilder::new(&mut robot, 0, &a_ra, a_bnd, b_bnd)
            .build_with_linearization()?;
        topp3_socp(&p, &clarabel)?
    };
    let (tf_sq1, t_s_sq1) = s_to_t_topp3(&s, &a_sq1, &b_sq1, num_sq1, 0.0);
    let st_sq1 = t_to_s_topp3(
        &s,
        &a_sq1,
        &b_sq1,
        num_sq1,
        &t_s_sq1,
        InterpolationMode::UniformTimeGrid(0.0, dt, true),
    );
    save3("topp3_socp_iter1", &a_sq1, &b_sq1, &t_s_sq1, &st_sq1)?;

    let (a_sq2, b_sq2, num_sq2) = {
        let p = Topp3ProblemBuilder::new(&mut robot, 0, &a_sq1, a_bnd, b_bnd)
            .build_with_linearization()?;
        topp3_socp(&p, &clarabel)?
    };
    let (tf_sq2, t_s_sq2) = s_to_t_topp3(&s, &a_sq2, &b_sq2, num_sq2, 0.0);
    let st_sq2 = t_to_s_topp3(
        &s,
        &a_sq2,
        &b_sq2,
        num_sq2,
        &t_s_sq2,
        InterpolationMode::UniformTimeGrid(0.0, dt, true),
    );
    save3("topp3_socp_iter2", &a_sq2, &b_sq2, &t_s_sq2, &st_sq2)?;
    println!("TOPP3-SOCP: iter1 = {tf_sq1:.4} s,  iter2 = {tf_sq2:.4} s");

    // ------------------------------------------------------------------
    // 6. COPP3-SOCP  (time + thermal-energy objective; SCP iterations 1 and 2)
    // ------------------------------------------------------------------
    let weights = vec![1.0f64; DIM];
    let copp_obj = [
        CoppObjective::Time(1.0),
        CoppObjective::ThermalEnergy(0.1, &weights),
    ];

    let (a_c1, b_c1, num_c1) = {
        let p = Copp3ProblemBuilder::new(&mut robot, &copp_obj, 0, &a_ra, a_bnd, b_bnd)
            .build_with_linearization()?;
        copp3_socp(&p, &clarabel)?
    };
    let (tf_c1, t_s_c1) = s_to_t_topp3(&s, &a_c1, &b_c1, num_c1, 0.0);
    let st_c1 = t_to_s_topp3(
        &s,
        &a_c1,
        &b_c1,
        num_c1,
        &t_s_c1,
        InterpolationMode::UniformTimeGrid(0.0, dt, true),
    );
    save3("copp3_socp_iter1", &a_c1, &b_c1, &t_s_c1, &st_c1)?;

    let (a_c2, b_c2, num_c2) = {
        let p = Copp3ProblemBuilder::new(&mut robot, &copp_obj, 0, &a_c1, a_bnd, b_bnd)
            .build_with_linearization()?;
        copp3_socp(&p, &clarabel)?
    };
    let (tf_c2, t_s_c2) = s_to_t_topp3(&s, &a_c2, &b_c2, num_c2, 0.0);
    let st_c2 = t_to_s_topp3(
        &s,
        &a_c2,
        &b_c2,
        num_c2,
        &t_s_c2,
        InterpolationMode::UniformTimeGrid(0.0, dt, true),
    );
    save3("copp3_socp_iter2", &a_c2, &b_c2, &t_s_c2, &st_c2)?;
    println!("COPP3-SOCP: iter1 = {tf_c1:.4} s,  iter2 = {tf_c2:.4} s");

    println!("\nAll data saved to output/data/");
    println!("Run each script in scripts/ to produce plots.");
    Ok(())
}
