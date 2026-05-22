//! Plots TOPP2-RA speed profile √a(s) and trajectory s(t).
//! Output: output/plot_topp2_ra.png

use std::f64::consts::PI;
use std::fs;
use copp::InterpolationMode;
use copp::path::{Jet3, Path, sin};
use copp::robot::Robot;
use copp::solver::topp2_ra::{
    ReachSet2OptionsBuilder, Topp2ProblemBuilder, s_to_t_topp2, t_to_s_topp2, topp2_ra,
};
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
    let a_ra = topp2_ra(&problem, &ReachSet2OptionsBuilder::new().build()?)?;

    let dt = 1e-3;
    let (t_final, t_s) = s_to_t_topp2(&s, &a_ra, 0.0);
    let s_t = t_to_s_topp2(
        &s,
        &a_ra,
        &t_s,
        InterpolationMode::UniformTimeGrid(0.0, dt, true),
    );
    println!("TOPP2-RA: t_final = {t_final:.4} s, samples = {}", s_t.len());

    // ---------- Plot ----------
    fs::create_dir_all("output")?;
    let root = BitMapBackend::new("output/plot_topp2_ra.png", (900, 620)).into_drawing_area();
    root.fill(&WHITE)?;
    let (top, bottom) = root.split_vertically(310);

    // Top: path speed ṡ(s) = √a(s)
    {
        let speed: Vec<f64> = a_ra.iter().map(|&a| a.sqrt()).collect();
        let ymax = speed.iter().cloned().fold(0.0f64, f64::max) * 1.15;

        let mut chart = ChartBuilder::on(&top)
            .caption("TOPP2-RA: Path Speed ṡ(s) = √a(s)", ("sans-serif", 18))
            .margin(12)
            .x_label_area_size(28)
            .y_label_area_size(52)
            .build_cartesian_2d(0.0f64..1.0f64, 0.0f64..ymax)?;
        chart
            .configure_mesh()
            .x_desc("s (path parameter)")
            .y_desc("ṡ (path speed)")
            .draw()?;
        chart
            .draw_series(LineSeries::new(
                s.iter().zip(speed.iter()).map(|(&x, &y)| (x, y)),
                BLUE.stroke_width(2),
            ))?
            .label("ṡ(s) = √a(s)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(2)));
        chart.configure_series_labels().border_style(BLACK).draw()?;
    }

    // Bottom: trajectory s(t)
    {
        let t_vec: Vec<f64> = (0..s_t.len()).map(|i| i as f64 * dt).collect();

        let mut chart = ChartBuilder::on(&bottom)
            .caption(
                format!("Trajectory s(t)  [t_final = {t_final:.3} s]"),
                ("sans-serif", 18),
            )
            .margin(12)
            .x_label_area_size(30)
            .y_label_area_size(52)
            .build_cartesian_2d(0.0f64..t_final, 0.0f64..1.05)?;
        chart
            .configure_mesh()
            .x_desc("t (s)")
            .y_desc("s(t)")
            .draw()?;
        chart
            .draw_series(LineSeries::new(
                t_vec.iter().zip(s_t.iter()).map(|(&t, &sv)| (t, sv)),
                RED.stroke_width(2),
            ))?
            .label("s(t)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));
        chart.configure_series_labels().border_style(BLACK).draw()?;
    }

    root.present()?;
    println!("Saved: output/plot_topp2_ra.png");
    Ok(())
}
