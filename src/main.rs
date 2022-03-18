use glam::Vec3;
use softbuffer::GraphicsContext;
use std::{sync::Arc, thread, time::Instant};
use threadpool::ThreadPool;
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const SAMPLES_PER_PIXEL: u32 = 64;
const MAX_BOUNCES: u32 = 8;
const BLOCK_SIZE: u32 = 32;

#[derive(Debug)]
pub enum RenderEvent {
    BlockComplete(u32, u32, u32, u32, Vec<u32>),
}

pub struct RandomSeries {
    state: u32,
}

impl RandomSeries {
    pub fn new(seed: u32) -> Self {
        RandomSeries { state: seed }
    }

    fn xorshift32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    pub fn random01(&mut self) -> f32 {
        (self.xorshift32() as f64 / u32::MAX as f64) as f32
    }

    pub fn random(&mut self, min: f32, max: f32) -> f32 {
        self.random01() * (max - min) + min
    }

    pub fn range_i32(&mut self, min: i32, max: i32) -> i32 {
        let diff = max - min;
        if diff > 0 {
            (self.xorshift32() % diff as u32) as i32 + min
        } else {
            min
        }
    }

    pub fn range_u32(&mut self, min: u32, max: u32) -> u32 {
        let diff = max - min;
        if diff > 0 {
            (self.xorshift32() % diff) + min
        } else {
            min
        }
    }

    pub fn random_vec3(&mut self) -> Vec3 {
        Vec3::new(self.random01(), self.random01(), self.random01())
    }

    pub fn random_in_unit_sphere(&mut self) -> Vec3 {
        loop {
            let p = self.random_vec3();
            if p.length_squared() < 1.0 {
                return p;
            }
        }
    }

    pub fn random_on_unit_sphere(&mut self) -> Vec3 {
        self.random_in_unit_sphere().normalize()
    }

    pub fn random_in_hemisphere(&mut self, normal: &Vec3) -> Vec3 {
        let v = self.random_in_unit_sphere();
        if Vec3::dot(v, *normal) > 0.0 {
            v
        } else {
            -v
        }
    }
}

pub fn reflect(v: &Vec3, n: &Vec3) -> Vec3 {
    *v - 2.0 * Vec3::dot(*v, *n) * *n
}

pub fn refract(uv: &Vec3, n: &Vec3, etai_over_etat: f32) -> Vec3 {
    let cos_theta = f32::min(Vec3::dot(-*uv, *n), 1.0);
    let r_out_perp = etai_over_etat * (*uv + cos_theta * *n);
    let r_out_parallel = -f32::sqrt(f32::abs(1.0 - r_out_perp.length_squared())) * *n;
    r_out_perp + r_out_parallel
}

pub fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    // Use Schlick's approximation for reflectance.
    let mut r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    r0 + (1.0 - r0) * f32::powf(1.0 - cosine, 5.0)
}

#[derive(Copy, Clone)]
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
    (1.0 - t) * white + t * blue
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

#[derive(Clone)]
pub struct Hit {
    pub p: Vec3,
    pub normal: Vec3,
    pub t: f32,
    pub front_face: bool,
    pub material: Option<Arc<dyn Material>>,
}

impl Hit {
    fn new() -> Self {
        Hit {
            p: Vec3::zero(),
            normal: Vec3::zero(),
            t: 0.0,
            front_face: true,
            material: None,
        }
    }

    fn set_face_normal(&mut self, ray: &Ray, outward_normal: &Vec3) {
        self.front_face = Vec3::dot(ray.dir, *outward_normal) < 0.0;
        self.normal = if self.front_face {
            *outward_normal
        } else {
            -*outward_normal
        };
    }

    fn set_material(&mut self, material: Arc<dyn Material>) {
        self.material = Some(material);
    }
}

#[derive(Copy, Clone)]
pub struct Camera {
    origin: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    lower_left_corner: Vec3,
}

impl Camera {
    fn new(origin: &Vec3, viewport_width: f32, viewport_height: f32, focal_length: f32) -> Self {
        let horizontal = Vec3::new(viewport_width, 0.0, 0.0);
        let vertical = Vec3::new(0.0, viewport_height, 0.0);
        Camera {
            origin: *origin,
            horizontal,
            vertical,
            lower_left_corner: *origin
                - horizontal / 2.0
                - vertical / 2.0
                - Vec3::new(0.0, 0.0, focal_length),
        }
    }

    fn get_ray(&self, u: f32, v: f32) -> Ray {
        Ray {
            origin: self.origin,
            dir: self.lower_left_corner + u * self.horizontal + v * self.vertical - self.origin,
        }
    }
}

pub trait Material: Sync + Send {
    fn scatter(
        &self,
        ray_in: &Ray,
        hit: &Hit,
        attenuation: &mut Vec3,
        ray_out: &mut Ray,
        series: &mut RandomSeries,
    ) -> bool;
}

pub struct LambertianMaterial {
    albedo: Vec3,
}

impl Material for LambertianMaterial {
    fn scatter(
        &self,
        _ray_in: &Ray,
        hit: &Hit,
        attenuation: &mut Vec3,
        ray_out: &mut Ray,
        series: &mut RandomSeries,
    ) -> bool {
        let mut d = hit.normal + series.random_on_unit_sphere();
        if d.x().abs() + d.y().abs() + d.z().abs() < 1e-8 {
            d = hit.normal;
        }
        *ray_out = Ray {
            origin: hit.p,
            dir: d.normalize(),
        };
        //let surface_color = albedo;
        let surface_color = (hit.normal + Vec3::new(1.0, 1.0, 1.0)) * 0.5;
        *attenuation = surface_color;
        true
    }
}

pub struct MetalMaterial {
    albedo: Vec3,
    fuzz: f32,
}

impl Material for MetalMaterial {
    fn scatter(
        &self,
        ray_in: &Ray,
        hit: &Hit,
        attenuation: &mut Vec3,
        ray_out: &mut Ray,
        series: &mut RandomSeries,
    ) -> bool {
        let d = reflect(&ray_in.dir, &hit.normal);
        *ray_out = Ray {
            origin: hit.p,
            dir: d + self.fuzz * series.random_in_unit_sphere(),
        };
        *attenuation = self.albedo;
        Vec3::dot(d, hit.normal) > 0.0
    }
}

pub struct DielectricMaterial {
    albedo: Vec3,
    ior: f32,
}

impl Material for DielectricMaterial {
    fn scatter(
        &self,
        ray_in: &Ray,
        hit: &Hit,
        attenuation: &mut Vec3,
        ray_out: &mut Ray,
        series: &mut RandomSeries,
    ) -> bool {
        *attenuation = self.albedo;
        let refraction_ratio = if hit.front_face {
            1.0 / self.ior
        } else {
            self.ior
        };

        let dir = ray_in.dir.normalize();
        let cos_theta = f32::min(Vec3::dot(-dir, hit.normal), 1.0);
        let sin_theta = f32::sqrt(1.0 - cos_theta * cos_theta);
        let cannot_refract = (refraction_ratio * sin_theta) > 1.0;
        let direction =
            if cannot_refract || reflectance(cos_theta, refraction_ratio) > series.random01() {
                reflect(&dir, &hit.normal)
            } else {
                refract(&dir, &hit.normal, refraction_ratio)
            };

        *ray_out = Ray {
            origin: hit.p,
            dir: direction,
        };
        true
    }
}

pub trait Hittable: Sync + Send {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, hit: &mut Hit) -> bool;
}

pub struct Sphere {
    radius: f32,
    center: Vec3,
    material: Arc<dyn Material>,
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, hit: &mut Hit) -> bool {
        let oc = ray.origin - self.center;
        let a = Vec3::length_squared(ray.dir);
        let half_b = Vec3::dot(oc, ray.dir);
        let c = Vec3::length_squared(oc) - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant < 0.0 {
            return false;
        }
        let sqrtd = discriminant.sqrt();

        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || t_max < root {
            root = (-half_b + sqrtd) / a;
            if root < t_min || t_max < root {
                return false;
            }
        }

        hit.t = root;
        hit.p = ray.at(hit.t);
        let outward_normal = (hit.p - self.center) / self.radius;
        hit.set_face_normal(ray, &outward_normal);
        hit.set_material(self.material.clone());
        true
    }
}

pub struct HittableList {
    objects: Vec<Box<dyn Hittable>>,
}

impl Hittable for HittableList {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, hit: &mut Hit) -> bool {
        let mut tmp_hit = Hit::new();
        let mut hit_anything = false;
        let mut closest_so_far = t_max;

        for object in self.objects.iter() {
            if object.hit(ray, t_min, closest_so_far, &mut tmp_hit) {
                hit_anything = true;
                closest_so_far = tmp_hit.t;
                *hit = tmp_hit.clone();
            }
        }

        hit_anything
    }
}

fn render(width: u32, height: u32, event_loop: &EventLoop<RenderEvent>, pool: &mut ThreadPool) {
    let aspect_ratio = width as f32 / height as f32;
    let viewport_height = 2.0;
    let viewport_width = aspect_ratio * viewport_height;
    let camera = Camera::new(
        &Vec3::new(0.0, 0.0, 0.0),
        viewport_width,
        viewport_height,
        1.0,
    );

    // World
    let material_center = Arc::new(LambertianMaterial {
        albedo: Vec3::one(),
    });
    let material_ground = Arc::new(LambertianMaterial {
        albedo: Vec3::one(),
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

    let world = Arc::new(HittableList {
        objects: vec![
            Box::new(Sphere {
                center: Vec3::new(0.0, -100.5, -1.0),
                radius: 100.0,
                material: material_ground.clone(),
            }),
            Box::new(Sphere {
                center: Vec3::new(-1.0, 0.0, -1.0),
                radius: 0.5,
                material: material_glass.clone(),
            }),
            Box::new(Sphere {
                center: Vec3::new(-1.0, 0.0, -1.0),
                radius: -0.45,
                material: material_glass.clone(),
            }),
            /*
            Box::new(Sphere {
                center: Vec3::new(-1.0, 0.0, -1.0),
                radius: 0.5,
                material: material_left.clone(),
            }),
            */
            Box::new(Sphere {
                center: Vec3::new(1.0, 0.0, -1.0),
                radius: 0.5,
                material: material_right.clone(),
            }),
            Box::new(Sphere {
                center: Vec3::new(0.0, 0.0, -1.0),
                radius: 0.5,
                material: material_center.clone(),
            }),
        ],
    });

    let num_vertical_blocks = (height as f32 / BLOCK_SIZE as f32).ceil() as u32;
    let num_horizontal_blocks = (width as f32 / BLOCK_SIZE as f32).ceil() as u32;
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

                let mut series = RandomSeries::new(j);
                let mut block = vec![0; (block_width * block_height) as usize];
                let ix = i * BLOCK_SIZE;
                let iy = j * BLOCK_SIZE;
                for y in 0..block_height {
                    for x in 0..block_width {
                        let mut accumulated_pixel_color = Vec3::zero();
                        for _ in 0..SAMPLES_PER_PIXEL {
                            let u = ((ix + x) as f32 + series.random01()) / (width - 1) as f32;
                            let v = ((iy + y) as f32 + series.random01()) / (height - 1) as f32;
                            let ray = camera.get_ray(u, 1.0 - v);
                            accumulated_pixel_color += ray_color(&ray, &*world, &mut series, 0);
                        }

                        const PIXEL_COLOR_SCALE: f32 = 1.0 / SAMPLES_PER_PIXEL as f32;
                        let c = accumulated_pixel_color * PIXEL_COLOR_SCALE;
                        let pixel_color = Vec3::new(c.x().sqrt(), c.y().sqrt(), c.z().sqrt());
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
    let mut pool = threadpool::Builder::new().build();
    render(width, height, &event_loop, &mut pool);

    let mut buffer = (0..((width * height) as usize))
        .map(|_| 0x00FFFFFF as u32)
        .collect::<Vec<_>>();
    graphics_context.set_buffer(&buffer, width as u16, height as u16);

    thread::spawn(move || {
        pool.join();
        let elapsed = now.elapsed();
        println!("Finished in {:.2?}", elapsed);
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
