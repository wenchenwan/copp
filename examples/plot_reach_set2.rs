//! Plots the TOPP2 reachable set bounds (a_max / a_min) and overlays the
//! TOPP2-RA optimal solution.
//! Output: output/plot_reach_set2.png

use std::f64::consts::PI;
use std::fs;
use copp::path::{Jet3, Path, sin};
use copp::robot::Robot;
use copp::solver::topp2_ra::{
    ReachSet2OptionsBuilder, Topp2ProblemBuilder, topp2_ra,
};
use copp::solver::reach_set2::{reach_set2_backward, reach_set2_bidirectional};
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const DIM: usize = 3;
    let n = 1001;
    let s: Vec<f64> = (0..n).map(|j| j as f64 / (n - 1) as f64).collect();

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
    let derivs = path.evaluate_up_to_2nd(&s)?;

    let vel_max = vec![1.0f64; DIM];
    let vel_min = vec![-1.0f64; DIM];
    let acc_max = vec![1.0f64; DIM];
    let acc_min = vec![-1.0f64; DIM];

    let mut robot = Robot::with_capacity(DIM, n);
    robot.with_s(&s)?;
    robot.with_q(
        &derivs.q.as_view(),
        &derivs.dq.as_ref().unwrap().as_view(),
        &derivs.ddq.as_ref().unwrap().as_view(),
        None,
        0,
    )?;
    robot.with_axial_velocity((vel_max.as_slice(), n), (vel_min.as_slice(), n), 0)?;
    robot.with_axial_acceleration((acc_max.as_slice(), n), (acc_min.as_slice(), n), 0)?;

    let problem = Topp2ProblemBuilder::new(&robot, (0, n - 1), (0.0, 0.0)).build()?;
    let options = ReachSet2OptionsBuilder::new().build()?;

    let reach_back = reach_set2_backward(&problem, &options)?;
    let reach_bidir = reach_set2_bidirectional(&problem, &options)?;
    let a_ra = topp2_ra(&problem, &options)?;

    println!(
        "reach_set2: backward a_max[mid]={:.4}, bidir a_max[mid]={:.4}, RA a[mid]={:.4}",
        reach_back.a_max[n / 2],
        reach_bidir.a_max[n / 2],
        a_ra[n / 2],
    );

    // ---------- Plot ----------
    fs::create_dir_all("output")?;
    let root =
        BitMapBackend::new("output/plot_reach_set2.png", (1000, 520)).into_drawing_area();
    root.fill(&WHITE)?;

    let ymax = reach_back.a_max.iter().cloned().fold(0.0f64, f64::max) * 1.08;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "TOPP2 Reachable Set: a(s) = ṡ²  (boundary: a_start = a_end = 0)",
            ("sans-serif", 18),
        )
        .margin(15)
        .x_label_area_size(32)
        .y_label_area_size(58)
        .build_cartesian_2d(0.0f64..1.0f64, 0.0f64..ymax)?;

    chart
        .configure_mesh()
        .x_desc("s (path parameter)")
        .y_desc("a(s) = ṡ²")
        .draw()?;

    // Backward-only upper bound (looser — only terminal boundary is constrained)
    chart
        .draw_series(LineSeries::new(
            s.iter()
                .zip(reach_back.a_max.iter())
                .map(|(&x, &y)| (x, y)),
            RGBColor(200, 140, 0).stroke_width(1),
        ))?
        .label("a_max  backward-only")
        .legend(|(x, y)| {
            PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(200, 140, 0).stroke_width(1))
        });

    // Bidirectional upper bound (tighter — both boundaries constrained)
    chart
        .draw_series(LineSeries::new(
            s.iter()
                .zip(reach_bidir.a_max.iter())
                .map(|(&x, &y)| (x, y)),
            RED.stroke_width(1),
        ))?
        .label("a_max  bidirectional")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(1)));

    // Bidirectional lower bound
    chart
        .draw_series(LineSeries::new(
            s.iter()
                .zip(reach_bidir.a_min.iter())
                .map(|(&x, &y)| (x, y)),
            BLUE.stroke_width(1),
        ))?
        .label("a_min  bidirectional")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(1)));

    // TOPP2-RA optimal solution (lies within bidir bounds)
    chart
        .draw_series(LineSeries::new(
            s.iter().zip(a_ra.iter()).map(|(&x, &y)| (x, y)),
            GREEN.stroke_width(2),
        ))?
        .label("TOPP2-RA  a(s)  [optimal]")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN.stroke_width(2)));

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.85))
        .draw()?;
    root.present()?;
    println!("Saved: output/plot_reach_set2.png");
    Ok(())
}
