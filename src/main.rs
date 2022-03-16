use glam::{Mat4, Vec3};
use softbuffer::GraphicsContext;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

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

fn ray_color(ray: &Ray, world: &dyn Hittable) -> Vec3 {
    let mut hit = Hit::new();
    if world.hit(ray, 0.0, f32::INFINITY, &mut hit) {
        return 0.5 * (hit.normal + Vec3::new(1.0, 1.0, 1.0));
    }
    sky_color(ray)
}

#[derive(Copy, Clone)]
pub struct Hit {
    p: Vec3,
    normal: Vec3,
    t: f32,
    front_face: bool,
}

impl Hit {
    fn new() -> Self {
        Hit {
            p: Vec3::zero(),
            normal: Vec3::zero(),
            t: 0.0,
            front_face: true,
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
}

pub trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, hit: &mut Hit) -> bool;
}

pub struct Sphere {
    radius: f32,
    center: Vec3,
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
                closest_so_far = hit.t;
                *hit = tmp_hit;
            }
        }

        hit_anything
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

    // World
    let mut world = HittableList {
        objects: vec![
            Box::new(Sphere {
                center: Vec3::new(0.0, 0.0, -1.0),
                radius: 0.5,
            }),
            Box::new(Sphere {
                center: Vec3::new(0.0, -100.5, -1.0),
                radius: 100.0,
            }),
        ],
    };

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
            let pixel_color = ray_color(&ray, &world);
            buffer[((height - j - 1) * width + i) as usize] = rgb_to_u32(pixel_color);
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
