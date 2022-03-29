mod camera;
mod material;
mod math;
mod shape;

use crate::camera::*;
use crate::material::*;
use crate::math::*;
use crate::shape::*;

use glam::Vec3;
use num_cpus;
use softbuffer::GraphicsContext;
use std::{sync::Arc, thread, time::Instant};
use threadpool::ThreadPool;
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const SAMPLES_PER_PIXEL: u32 = 50;
const MAX_BOUNCES: u32 = 5;
const BLOCK_SIZE: u32 = 16;
const NUM_THREADS: usize = 0;

#[derive(Debug)]
pub enum RenderEvent {
    BlockComplete(u32, u32, u32, u32, Vec<u32>),
}

fn sky_color(ray: &Ray) -> Vec3 {
    let dir = Vec3::normalize(ray.dir);
    let t = 0.5 * (dir.y() + 1.0);
    let white = Vec3::new(1.0, 1.0, 1.0);
    let blue = Vec3::new(0.5, 0.7, 1.0);

    let sky_intensity = Vec3::dot(dir, Vec3::new(0.0, 0.0, 1.0)).max(0.7);

    ((1.0 - t) * white + t * blue) * sky_intensity
}

fn ray_color(ray: &Ray, world: &dyn Hittable, series: &mut RandomSeries, depth: u32) -> Vec3 {
    if depth >= MAX_BOUNCES {
        return Vec3::zero();
    }

    let mut hit = Hit::new();
    if world.hit(ray, 0.001, f32::INFINITY, &mut hit) {
        let mut bounce_ray = Ray {
            origin: Vec3::zero(),
            dir: Vec3::zero(),
        };
        let mut attenuation = Vec3::zero();
        if hit.material.as_ref().unwrap().scatter(
            ray,
            &hit,
            &mut attenuation,
            &mut bounce_ray,
            series,
        ) {
            attenuation * ray_color(&bounce_ray, world, series, depth + 1)
        } else {
            Vec3::zero()
        }
    } else {
        sky_color(ray)
    }
}

fn render(width: u32, height: u32, event_loop: &EventLoop<RenderEvent>, pool: &mut ThreadPool) {
    let aspect_ratio = width as f32 / height as f32;
    let camera = Camera::new(
        Vec3::new(0.0, -2.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::unit_z(),
        88.0,
        aspect_ratio,
        0.2,
        0.75,
    );
    /*
    let camera = Camera::new(
        Vec3::new(0.0, -3.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::unit_z(),
        90.0,
        aspect_ratio,
    );
    */
    /*
    let camera = Camera::new(
        Vec3::new(2.0, -3.0, 1.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::unit_z(),
        60.0,
        aspect_ratio,
    );
    */

    // World
    let material_center = Arc::new(LambertianMaterial {
        albedo: Vec3::one(),
        normal: true,
    });
    let material_ground = Arc::new(LambertianMaterial {
        albedo: Vec3::new(0.13, 0.8, 0.13),
        normal: false,
    });
    let material_left = Arc::new(MetalMaterial {
        albedo: Vec3::splat(0.8),
        fuzz: 0.3,
    });
    let material_right = Arc::new(MetalMaterial {
        albedo: Vec3::splat(0.4),
        fuzz: 1.0,
    });
    let material_glass = Arc::new(DielectricMaterial {
        albedo: Vec3::splat(1.0),
        ior: 1.3,
    });

    let mut world = HittableList {
        objects: vec![
            Box::new(Sphere {
                center: Vec3::new(0.0, -1.0, -200.5),
                radius: 200.0,
                material: material_ground.clone(),
            }),
            Box::new(Sphere {
                center: Vec3::new(-1.0, -1.0, 0.0),
                radius: 0.5,
                material: material_glass.clone(),
            }),
            /*
            Box::new(Sphere {
                center: Vec3::new(-1.0, -1.0, 0.0),
                radius: -0.45,
                material: material_glass.clone(),
            }),
            */
            Box::new(Sphere {
                center: Vec3::new(1.0, -1.0, 0.0),
                radius: 0.5,
                material: material_right.clone(),
            }),
            Box::new(Sphere {
                center: Vec3::new(0.0, -1.0, 0.0),
                radius: 0.5,
                material: material_center.clone(),
            }),
            Box::new(Sphere {
                center: Vec3::new(-0.4, -1.3, -0.5 + 0.1),
                radius: 0.1,
                material: material_right.clone(),
            }),
            Box::new(Sphere {
                center: Vec3::new(0.4, -1.3, -0.5 + 0.1),
                radius: 0.1,
                material: material_left.clone(),
            }),
        ],
    };

    /*
    let mut r = RandomSeries::new(8);
    let materials: Vec<Arc<dyn Material>> = vec![
        material_center,
        material_left,
        material_right,
        material_glass,
    ];
    for i in 0..12 {
        let p = Vec3::new(r.random(-3.0, 3.0), r.random(-3.0, 3.0), -0.25);
        world.objects.push(Box::new(Sphere {
            center: p,
            radius: 0.25,
            material: materials[r.range_i32(0, materials.len() as i32) as usize].clone(),
        }));
    }
    */

    let world = Arc::new(world);

    let num_vertical_blocks = (height as f32 / BLOCK_SIZE as f32).ceil() as u32;
    let num_horizontal_blocks = (width as f32 / BLOCK_SIZE as f32).ceil() as u32;
    let one_over_width = 1.0 / (width - 1) as f32;
    let one_over_height = 1.0 / (height - 1) as f32;
    for j in 0..num_vertical_blocks {
        for i in 0..num_horizontal_blocks {
            let event_loop_proxy = event_loop.create_proxy();
            let world = world.clone();
            pool.execute(move || {
                let block_width = if (i + 1) * BLOCK_SIZE > width {
                    BLOCK_SIZE - ((i + 1) * BLOCK_SIZE - width)
                } else {
                    BLOCK_SIZE
                };
                let block_height = if (j + 1) * BLOCK_SIZE > height {
                    BLOCK_SIZE - ((j + 1) * BLOCK_SIZE - height)
                } else {
                    BLOCK_SIZE
                };

                let mut series = RandomSeries::new(j + 1);
                let mut block = vec![0; (block_width * block_height) as usize];
                let ix = i * BLOCK_SIZE;
                let iy = j * BLOCK_SIZE;
                for y in 0..block_height {
                    for x in 0..block_width {
                        let mut accumulated_pixel_color = Vec3::zero();
                        for _ in 0..SAMPLES_PER_PIXEL {
                            let u = ((ix + x) as f32 + series.random01()) * one_over_width;
                            let v = ((iy + y) as f32 + series.random01()) * one_over_height;
                            let ray = camera.get_ray(u, 1.0 - v, &mut series);
                            accumulated_pixel_color += ray_color(&ray, &*world, &mut series, 0);
                        }

                        const PIXEL_COLOR_SCALE: f32 = 1.0 / SAMPLES_PER_PIXEL as f32;
                        let c = accumulated_pixel_color * PIXEL_COLOR_SCALE;
                        let pixel_color =
                            Vec3::new(c.x().sqrt(), c.y().sqrt(), c.z().sqrt()).min(Vec3::one());
                        block[(y * block_width + x) as usize] = rgb_to_u32(pixel_color);
                    }
                }
                event_loop_proxy
                    .send_event(RenderEvent::BlockComplete(
                        ix,
                        iy,
                        block_width,
                        block_height,
                        block,
                    ))
                    .unwrap();
            });
        }
    }
}

fn main() {
    let window_width = 800.0;
    let window_height = 600.0;
    let event_loop = EventLoop::<RenderEvent>::with_user_event();
    let window = WindowBuilder::new()
        .with_resizable(false)
        //.with_decorations(false)
        .with_inner_size(LogicalSize::new(window_width, window_height))
        .build(&event_loop)
        .unwrap();
    let mut graphics_context = unsafe { GraphicsContext::new(window) }.unwrap();

    let (width, height) = {
        let size = graphics_context.window().inner_size();
        (size.width, size.height)
    };

    let now = Instant::now();
    let num_threads = if NUM_THREADS > 0 {
        NUM_THREADS
    } else {
        num_cpus::get()
    };
    println!("Creating threadpool with {num_threads} threads.");
    let mut pool = ThreadPool::new(num_threads);
    render(width, height, &event_loop, &mut pool);

    let mut buffer = (0..((width * height) as usize))
        .map(|_| 0x00FFFFFF as u32)
        .collect::<Vec<_>>();
    graphics_context.set_buffer(&buffer, width as u16, height as u16);

    thread::spawn(move || {
        pool.join();
        let elapsed = now.elapsed();
        println!("Finished in {:.2?}.", elapsed);
    });

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
            Event::UserEvent(event) => match event {
                RenderEvent::BlockComplete(x, y, bw, bh, data) => {
                    for j in 0..bh {
                        for i in 0..bw {
                            /*
                            unsafe {
                                *buffer.get_unchecked_mut(((j + y) * width + i + x) as usize) =
                                    *data.get_unchecked((j * bw + i) as usize);
                            }
                            */
                            buffer[((j + y) * width + i + x) as usize] =
                                data[(j * bw + i) as usize];
                        }
                    }
                    graphics_context.set_buffer(&buffer, width as u16, height as u16);
                }
            },
            _ => {}
        }
    });
}
