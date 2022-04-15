use crate::material::*;
use crate::math::*;
use glam::{Mat4, Vec3};
use std::sync::Arc;

#[derive(Copy, Clone)]
pub struct Camera {
    // params
    look_from: Vec3,
    look_at: Vec3,
    vup: Vec3,
    vfov: f32,
    aperture: f32,
    focus_dist: f32,

    // setup
    origin: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    lower_left_corner: Vec3,
    lens_radius: f32,
    w: Vec3,
    u: Vec3,
    v: Vec3,
}

impl Camera {
    pub fn new(
        look_from: Vec3,
        look_at: Vec3,
        vup: Vec3,
        vfov: f32,
        aperture: f32,
        focus_dist: f32,
    ) -> Self {
        Camera {
            look_from,
            look_at,
            vup,
            vfov,
            aperture,
            focus_dist,

            origin: Vec3::zero(),
            horizontal: Vec3::zero(),
            vertical: Vec3::zero(),
            lower_left_corner: Vec3::zero(),
            lens_radius: 0.0,
            w: Vec3::zero(),
            u: Vec3::zero(),
            v: Vec3::zero(),
        }
    }

    pub fn setup(&mut self, aspect_ratio: f32) {
        let theta = self.vfov * std::f32::consts::PI / 180.0;
        let h = f32::tan(theta / 2.0);
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (self.look_from - self.look_at).normalize();
        let u = self.vup.cross(w).normalize();
        let v = w.cross(u);

        let origin = self.look_from;
        let horizontal = self.focus_dist * viewport_width * u;
        let vertical = self.focus_dist * viewport_height * v;
        let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - self.focus_dist * w;

        self.origin = origin;
        self.horizontal = horizontal;
        self.vertical = vertical;
        self.lower_left_corner = lower_left_corner;
        self.lens_radius = self.aperture / 2.0;
        self.w = w;
        self.u = u;
        self.v = v;
    }

    pub fn get_ray(&self, s: f32, t: f32, r: &mut RandomSeries) -> Ray {
        let rd = self.lens_radius * r.random_in_disk();
        let offset = self.u * rd.x() + self.v * rd.y();
        Ray {
            origin: self.origin + offset,
            dir: self.lower_left_corner + s * self.horizontal + t * self.vertical
                - self.origin
                - offset,
        }
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
    pub fn new() -> Self {
        Hit {
            p: Vec3::zero(),
            normal: Vec3::zero(),
            t: 0.0,
            front_face: true,
            material: None,
        }
    }

    pub fn set_face_normal(&mut self, ray: &Ray, outward_normal: &Vec3) {
        self.front_face = Vec3::dot(ray.dir, *outward_normal) < 0.0;
        self.normal = if self.front_face {
            *outward_normal
        } else {
            -*outward_normal
        };
    }

    pub fn set_material(&mut self, material: Arc<dyn Material>) {
        self.material = Some(material);
    }
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

    pub fn transform(&self, t: &Mat4) -> Self {
        Ray {
            origin: t.transform_point3(self.origin),
            dir: t.transform_vector3(self.dir),
        }
    }
}
