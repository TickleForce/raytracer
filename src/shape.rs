use crate::camera::*;
use crate::material::*;
use glam::Vec3;
use std::{sync::Arc};

pub trait Hittable: Sync + Send {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, hit: &mut Hit) -> bool;
}

pub struct Sphere {
    pub radius: f32,
    pub center: Vec3,
    pub material: Arc<dyn Material>,
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
    pub objects: Vec<Box<dyn Hittable>>,
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
