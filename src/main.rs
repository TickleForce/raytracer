use glam::{Mat4, Vec3};
use softbuffer::GraphicsContext;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
//use std::num::Float;

pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

impl Ray {
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

fn rgb_to_u32(col: Vec3) -> u32 {
    ((255.999 * col.x()) as u32) << 16
        | ((255.999 * col.y()) as u32) << 8
        | ((255.999 * col.z()) as u32)
}

fn sky_color(ray: &Ray) -> Vec3 {
    let dir = Vec3::normalize(ray.dir);
    let t = 0.5 * (dir.y() + 1.0);
    let white = Vec3::new(1.0, 1.0, 1.0);
    let blue = Vec3::new(0.5, 0.7, 1.0);
    (1.0 - t) * blue + t * white
}

fn hit_sphere(center: &Vec3, radius: f32, ray: &Ray) -> f32 {
    let oc = ray.origin - *center;
    let a = Vec3::length_squared(ray.dir);
    let half_b = Vec3::dot(oc, ray.dir);
    let c = Vec3::length_squared(oc) - radius * radius;
    let discriminant = half_b * half_b - a * c;

    if discriminant < 0.0 {
        return -1.0;
    }
    (-half_b - discriminant.sqrt()) / a
}

fn ray_color(ray: &Ray) -> Vec3 {
    let t = hit_sphere(&Vec3::new(0.0, 0.0, -1.0), 0.5, ray);
    if t > 0.0 {
        let normal = ray.at(t).normalize() - Vec3::new(0.0, 0.0, -1.0);
        return 0.5 * Vec3::new(normal.x() + 1.0, normal.y() + 1.0, normal.z() + 1.0);
    }
    sky_color(ray)
}

pub struct Hit {
    p: Vec3,
    normal: Vec3,
    t: f32,
}

pub trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, hit: &Hit) -> bool;
}

pub struct Sphere {
    radius: f32,
    center: Vec3,
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, hit: &Hit) -> bool {
        return false;
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_resizable(false)
        //.with_decorations(false)
        .build(&event_loop)
        .unwrap();
    let mut graphics_context = unsafe { GraphicsContext::new(window) }.unwrap();

    let (width, height) = {
        let size = graphics_context.window().inner_size();
        (size.width, size.height)
    };
    let aspect_ratio = width as f32 / height as f32;
    let mut buffer = (0..((width * height) as usize))
        .map(|_| 0x00FFFFFF as u32)
        .collect::<Vec<_>>();

    // Camera
    let viewport_height = 2.0;
    let viewport_width = aspect_ratio * viewport_height;
    let focal_length = 1.0;

    let origin = Vec3::new(0.0, 0.0, 0.0);
    let horizontal = Vec3::new(viewport_width, 0.0, 0.0);
    let vertical = Vec3::new(0.0, viewport_height, 0.0);
    let lower_left_corner =
        origin - horizontal / 2.0 - vertical / 2.0 - Vec3::new(0.0, 0.0, focal_length);

    println!();
    for j in 0..height {
        let row = j + 1;
        print!("\rRow: {row}/{height}");
        for i in 0..width {
            let u = i as f32 / (width - 1) as f32;
            let v = j as f32 / (height - 1) as f32;
            let ray = Ray {
                origin,
                dir: lower_left_corner + u * horizontal + v * vertical - origin,
            };

            let pixel_color = ray_color(&ray);

            /*
            let col = Vec3::new(
                (i as f32) / (width as f32),
                (j as f32) / (height as f32),
                0.25,
            );
            buffer[(j * width + i) as usize] = rgb_to_u32(col);
            */
            buffer[(j * width + i) as usize] = rgb_to_u32(pixel_color);
        }
    }
    println!();

    graphics_context.set_buffer(&buffer, width as u16, height as u16);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::RedrawRequested(window_id) if window_id == graphics_context.window().id() => {}
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id,
            } if window_id == graphics_context.window().id() => {
                *control_flow = ControlFlow::Exit;
            }
            _ => {}
        }
    });
}
