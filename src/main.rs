mod camera;
mod material;
mod math;
mod shape;

use crate::{camera::*, material::*, math::*, shape::*};

use glam::{vec3, Mat4, Vec3};
use num_cpus;
use softbuffer::GraphicsContext;
use std::{collections::HashMap, sync::Arc, thread, time::Instant};
use threadpool::ThreadPool;
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const SAMPLES_PER_PIXEL: u32 = 400;
const MAX_BOUNCES: u32 = 6;
const BLOCK_SIZE: u32 = 64;
const NUM_THREADS: usize = 10;
const IMAGE_SIZE: f64 = 400.0;

#[derive(Debug)]
pub enum RenderEvent {
    BlockComplete(u32, u32, u32, u32, Vec<u32>),
}

fn sky_color(ray: &Ray, sky_intensity: f32) -> Vec3 {
    let dir = Vec3::normalize(ray.dir);
    let t = 0.5 * (dir.y() + 1.0);
    let white = Vec3::new(1.0, 1.0, 1.0);
    let blue = Vec3::new(0.5, 0.7, 1.0);

    ((1.0 - t) * white + t * blue) * sky_intensity
}

fn ray_color(ray: &Ray, world: &World, series: &mut RandomSeries, depth: u32) -> Vec3 {
    if depth >= MAX_BOUNCES {
        return Vec3::zero();
    }

    let mut hit = Hit::new();
    if world.hittable.hit(ray, 0.001, f32::INFINITY, &mut hit) {
        let mut bounce_ray = Ray {
            origin: Vec3::zero(),
            dir: Vec3::zero(),
        };
        let emitted = hit.material.as_ref().unwrap().emit(&hit);
        let mut attenuation = Vec3::zero();
        if hit.material.as_ref().unwrap().scatter(
            ray,
            &hit,
            &mut attenuation,
            &mut bounce_ray,
            series,
        ) {
            emitted + attenuation * ray_color(&bounce_ray, world, series, depth + 1)
        } else {
            emitted
        }
    } else {
        sky_color(ray, world.sky_intensity)
    }
}

struct World {
    camera: Camera,
    hittable: Box<dyn Hittable>,
    sky_intensity: f32,
}

fn create_world1() -> World {
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
    let material_glow = Arc::new(EmissiveMaterial {
        color: Vec3::splat(10.0),
    });

    let mut objects: Vec<Box<dyn Hittable>> = vec![
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
    ];

    let mut r = RandomSeries::new(10);
    let materials: Vec<Arc<dyn Material>> = vec![
        material_center,
        material_left,
        material_right,
        material_glass,
    ];
    for _i in 0..500 {
        //let p = Vec3::new(r.random(-5.0, 5.0), r.random(-5.0, 5.0), -0.25);
        let p = Vec3::new(
            r.random(-5.0, 5.0),
            r.random(-5.0, 5.0),
            r.random(-0.25, 5.0),
        );
        objects.push(Box::new(Sphere {
            center: p,
            radius: 0.25,
            material: materials[r.range_i32(0, materials.len() as i32) as usize].clone(),
        }));
    }

    objects.push(Box::new(Sphere {
        center: Vec3::new(0.0, 0.0, 50.0),
        radius: 12.0,
        material: material_glow.clone(),
    }));

    let camera = Camera::new(
        Vec3::new(2.0, -3.0, 1.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::unit_z(),
        60.0,
        0.0,
        10.0,
    );

    //let world = Box::new(HittableList { objects });
    let hittable = Box::new(Bvh::new(objects, 0));
    World {
        camera,
        hittable,
        sky_intensity: 0.02,
    }
}

fn create_world2() -> World {
    let material1: Arc<dyn material::Material + 'static> = Arc::new(LambertianMaterial {
        albedo: Vec3::one(),
        normal: false,
    });
    /*
    let material2 = Arc::new(MetalMaterial {
        albedo: Vec3::splat(0.8),
        fuzz: 0.2,
    });
    */
    let material_glow = Arc::new(EmissiveMaterial {
        color: Vec3::splat(4.0),
    });

    let objects: Vec<Box<dyn Hittable>> = vec![
        /*
        Box::new(Mesh::plane(
            2.0, 2.0,
            material1.clone(),
            Mat4::from_translation(vec3(0.0, 0.0, -0.5)),
        )),
        Box::new(Mesh::plane(
            2.0, 2.0,
            material1.clone(),
            Mat4::from_translation(vec3(0.0, 0.0, 1.0)),
        )),
        */
        Box::new(Mesh::from_file(
            "thing.obj",
            HashMap::from([("default".to_string(), material1.clone())]),
            Mat4::identity(),
        )),
        /*
        Box::new(Sphere {
            center: Vec3::new(0.0, 0.0, 0.25),
            radius: 0.5,
            material: material1.clone(),
        }),
        */
        Box::new(Sphere {
            center: Vec3::new(0.0, 0.0, 1.5),
            radius: 0.65,
            material: material_glow.clone(),
        }),
    ];

    let camera = Camera::new(
        Vec3::new(0.0, -2.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::unit_z(),
        88.0,
        0.001,
        1.0,
    );

    let hittable = Box::new(Bvh::new(objects, 0));
    //let hittable = Box::new(HittableList { objects });
    World {
        camera,
        hittable,
        sky_intensity: 1.5,
    }
}

fn create_cornell_box(filename: &str) -> World {
    let mred: Arc<dyn material::Material + 'static> = Arc::new(LambertianMaterial {
        albedo: vec3(0.65, 0.05, 0.05),
        normal: false,
    });
    /*
    let mred: Arc<dyn material::Material + 'static> = Arc::new(MetalMaterial {
        albedo: vec3(0.65, 0.05, 0.05),
        fuzz: 12.0,
    });
    */
    let mwhite: Arc<dyn material::Material + 'static> = Arc::new(LambertianMaterial {
        albedo: vec3(0.73, 0.73, 0.73),
        normal: false,
    });
    /*
    let mwhite: Arc<dyn material::Material + 'static> = Arc::new(MetalMaterial {
        albedo: vec3(0.73, 0.73, 0.73),
        fuzz: 8.0,
    });
    */
    let mgreen: Arc<dyn material::Material + 'static> = Arc::new(LambertianMaterial {
        albedo: vec3(0.12, 0.45, 0.15),
        normal: false,
    });
    /*
    let mgreen: Arc<dyn material::Material + 'static> = Arc::new(MetalMaterial {
        albedo: vec3(0.12, 0.45, 0.15),
        fuzz: 12.0,
    });
    */
    let mlight: Arc<dyn material::Material + 'static> = Arc::new(EmissiveMaterial {
        color: vec3(10.0, 10.0, 10.0),
    });
    let mdefault: Arc<dyn material::Material + 'static> = Arc::new(LambertianMaterial {
        albedo: vec3(0.5, 0.5, 0.5),
        normal: false,
    });
    let mglass = Arc::new(DielectricMaterial {
        albedo: vec3(1.0, 1.0, 1.0),
        ior: 2.3,
    });
    let mmetal = Arc::new(MetalMaterial {
        albedo: Vec3::splat(0.8),
        fuzz: 3.6,
    });
    let hittable = Box::new(Mesh::from_file(
        filename,
        HashMap::from([
            ("default".to_string(), mdefault.clone()),
            ("red".to_string(), mred.clone()),
            ("green".to_string(), mgreen.clone()),
            ("white".to_string(), mwhite.clone()),
            ("light".to_string(), mlight.clone()),
            ("glass".to_string(), mglass.clone()),
            ("metal".to_string(), mmetal.clone()),
        ]),
        Mat4::identity(),
    ));

    let camera = Camera::new(
        vec3(760.0, -279.5, 274.5),
        vec3(0.0, -279.5, 274.5),
        vec3(0.0, 0.0, 1.0),
        40.0,
        0.0,
        200.0,
    );

    World {
        camera,
        hittable,
        sky_intensity: 0.5,
    }
}

fn render(width: u32, height: u32, event_loop: &EventLoop<RenderEvent>, pool: &mut ThreadPool) {
    //let mut world = create_world1();
    //let mut world = create_world2();
    let mut world = create_cornell_box("cornell_box.obj");
    //let mut world = create_cornell_box("cornell_box_plane.obj");
    let aspect_ratio = width as f32 / height as f32;
    world.camera.setup(aspect_ratio);

    let world = Arc::new(world);
    let num_vertical_blocks = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let num_horizontal_blocks = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
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
                let mut block_accumulater =
                    vec![Vec3::zero(); (block_width * block_height) as usize];
                let ix = i * BLOCK_SIZE;
                let iy = j * BLOCK_SIZE;

                let mut sample_count = 0;
                let mut next_sample_count = 10.min(SAMPLES_PER_PIXEL);
                loop {
                    for y in 0..block_height {
                        for x in 0..block_width {
                            let pixel_index = (y * block_width + x) as usize;
                            for _ in sample_count..next_sample_count {
                                let u = ((ix + x) as f32 + series.random01()) * one_over_width;
                                let v = ((iy + y) as f32 + series.random01()) * one_over_height;
                                let ray = world.camera.get_ray(u, 1.0 - v, &mut series);
                                block_accumulater[pixel_index] +=
                                    ray_color(&ray, &world, &mut series, 0);
                            }

                            let pixel_color_scale = 1.0 / next_sample_count as f32;
                            let c = block_accumulater[pixel_index] * pixel_color_scale;
                            let pixel_color = Vec3::new(c.x().sqrt(), c.y().sqrt(), c.z().sqrt())
                                .min(Vec3::one());
                            block[pixel_index] = rgb_to_u32(pixel_color);
                        }
                    }

                    // add a border to indicate in-progress tiles
                    if next_sample_count < SAMPLES_PER_PIXEL {
                        let border_color = rgb_to_u32(Vec3::new(1.0, 0.0, 0.0));
                        let mut set_pixel = |x: u32, y: u32, color: u32| {
                            block[(y * block_width + x) as usize] = color;
                        };
                        for x in 0..5.min(block_width) {
                            set_pixel(x, 0, border_color);
                            set_pixel(x, block_height - 1, border_color);
                            set_pixel(block_width - x - 1, 0, border_color);
                            set_pixel(block_width - x - 1, block_height - 1, border_color);
                        }
                        for y in 0..5.min(block_height) {
                            set_pixel(0, y, border_color);
                            set_pixel(block_width - 1, y, border_color);
                            set_pixel(0, block_height - y - 1, border_color);
                            set_pixel(block_width - 1, block_height - y - 1, border_color);
                        }
                    }

                    event_loop_proxy
                        .send_event(RenderEvent::BlockComplete(
                            ix,
                            iy,
                            block_width,
                            block_height,
                            block.clone(),
                        ))
                        .unwrap();

                    if next_sample_count >= SAMPLES_PER_PIXEL {
                        break;
                    }
                    sample_count = next_sample_count;
                    next_sample_count = (next_sample_count * 2).min(SAMPLES_PER_PIXEL);
                }
            });
        }
    }
}

fn main() {
    let window_width = IMAGE_SIZE;
    let window_height = IMAGE_SIZE;
    let event_loop = EventLoop::<RenderEvent>::with_user_event();
    let window = WindowBuilder::new()
        .with_resizable(false)
        //.with_decorations(false)
        .with_inner_size(LogicalSize::new(window_width, window_height))
        .with_title("Raytracer")
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
