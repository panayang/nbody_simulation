use serde::{Deserialize, Serialize};
use glam::DVec3;
use rayon::prelude::*;
use plotters::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::error::Error;
use indicatif::{ProgressBar, ProgressStyle};

// 万有引力常数 (单位: m^3 kg^-1 s^-2)
const G: f64 = 6.67430e-11;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Body {
    mass: f64,
    #[serde(with = "dvec3_serde")]
    position: DVec3,
    #[serde(with = "dvec3_serde")]
    velocity: DVec3,
    #[serde(skip)]
    acceleration: DVec3,
}

// 自定义 DVec3 的序列化/反序列化
mod dvec3_serde {
    use super::DVec3;
    use serde::{self, Deserialize, Deserializer, Serializer, Serialize};

    pub fn serialize<S>(vec: &DVec3, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        vec.to_array().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<DVec3, D::Error>
    where
        D: Deserializer<'de>,
    {
        let arr = <[f64; 3]>::deserialize(deserializer)?;
        Ok(DVec3::from_array(arr))
    }
}

impl Body {
    fn new(mass: f64, position: DVec3, velocity: DVec3) -> Self {
        Self { mass, position, velocity, acceleration: DVec3::ZERO }
    }
}

// 计算引力并更新加速度
fn update_accelerations(bodies: &mut [Body], softening_factor: f64) {
    let softening_sq = softening_factor * softening_factor;
    let positions_masses: Vec<_> = bodies.iter().map(|b| (b.position, b.mass)).collect();

    // 使用 Rayon 并行计算
    bodies.par_iter_mut().for_each(|body_i| {
        let mut total_acceleration = DVec3::ZERO;
        for (pos_j, mass_j) in &positions_masses {
            if body_i.position == *pos_j {
                continue;
            }
            let direction = *pos_j - body_i.position;
            let distance_sq = direction.length_squared();
            let force_magnitude = (G * mass_j) / (distance_sq + softening_sq);
            total_acceleration += direction.normalize() * force_magnitude;
        }
        body_i.acceleration = total_acceleration;
    });
}

// Leapfrog 积分法 (kick-drift-kick)
fn leapfrog_integrator(bodies: &mut [Body], dt: f64) {
    // Kick (半步)
    for body in bodies.iter_mut() {
        body.velocity += body.acceleration * (dt / 2.0);
    }

    // Drift (全步)
    for body in bodies.iter_mut() {
        body.position += body.velocity * dt;
    }

    // Kick (另半步) - 需要重新计算加速度
    // 在主循环中完成
}

// 绘制密度投影图
fn plot_density_projection(
    bodies: &[Body],
    axis1: char,
    axis2: char,
    file_name: &str,
    time_step: usize,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(file_name, (1024, 768)).into_drawing_area();
    root.fill(&BLACK)?;

    // 自动确定边界
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for body in bodies {
        let (p1, p2) = match (axis1, axis2) {
            ('x', 'y') => (body.position.x, body.position.y),
            ('x', 'z') => (body.position.x, body.position.z),
            ('y', 'z') => (body.position.y, body.position.z),
            _ => panic!("Invalid axes"),
        };
        min_x = min_x.min(p1);
        max_x = max_x.max(p1);
        min_y = min_y.min(p2);
        max_y = max_y.max(p2);
    }

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Density Projection ({}-{}) at t={}", axis1, axis2, time_step), ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(
        bodies.iter().map(|body| {
            let (p1, p2) = match (axis1, axis2) {
                ('x', 'y') => (body.position.x, body.position.y),
                ('x', 'z') => (body.position.x, body.position.z),
                ('y', 'z') => (body.position.y, body.position.z),
                _ => (0.0, 0.0),
            };
            Circle::new((p1, p2), 2, WHITE.filled())
        })
    )?;

    root.present()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // --- 参数设置 ---
    let input_file = "particles.json";
    let time_steps = 1000; // 总时间步数
    let dt = 1.0e3; // 每个时间步的长度 (s)
    let softening_factor = 1.0e3; // 软化因子，防止奇点，可调
    let plot_interval = 10; // 每隔多少步输出一次图像

    // --- 读取初始条件 ---
    println!("Reading initial conditions from '{}'...", input_file);
    let file = File::open(input_file)?;
    let reader = BufReader::new(file);
    let mut bodies: Vec<Body> = serde_json::from_reader(reader)?;
    println!("Successfully loaded {} bodies.", bodies.len());

    // 创建输出目录
    std::fs::create_dir_all("output")?;

    // --- 主循环 ---
    println!("Starting simulation...");
    let pb = ProgressBar::new(time_steps as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
        .progress_chars("##-"));

    // 初始加速度
    update_accelerations(&mut bodies, softening_factor);

    for i in 0..time_steps {
        // Leapfrog: Kick-Drift-Kick
        // 1. Kick (半步)
        for body in bodies.iter_mut() {
            body.velocity += body.acceleration * (dt / 2.0);
        }

        // 2. Drift (全步)
        for body in bodies.iter_mut() {
            body.position += body.velocity * dt;
        }

        // 3. 更新加速度
        update_accelerations(&mut bodies, softening_factor);

        // 4. Kick (另半步)
        for body in bodies.iter_mut() {
            body.velocity += body.acceleration * (dt / 2.0);
        }

        // --- 输出图像 ---
        if i % plot_interval == 0 {
            plot_density_projection(&bodies, 'x', 'y', &format!("output/xy_proj_{:04}.png", i), i)?;
            plot_density_projection(&bodies, 'x', 'z', &format!("output/xz_proj_{:04}.png", i), i)?;
            plot_density_projection(&bodies, 'y', 'z', &format!("output/yz_proj_{:04}.png", i), i)?;
        }

        pb.inc(1);
    }

    pb.finish_with_message("Simulation complete.");

    Ok(())
}
