use crate::camera::*;
use crate::material::*;
use crate::math::*;
use glam::Vec3;
use std::sync::Arc;

pub trait Hittable: Sync + Send {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, hit: &mut Hit) -> bool;
    fn get_aabb(&self) -> Aabb;
}

pub struct Sphere {
    pub radius: f32,
    pub center: Vec3,
    pub material: Arc<dyn Material>,
}

impl Hittable for Sphere {
    // TODO: return Option<Hit> instead of a bool
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

    fn get_aabb(&self) -> Aabb {
        Aabb {
            min: self.center - Vec3::splat(self.radius),
            max: self.center + Vec3::splat(self.radius),
        }
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

    fn get_aabb(&self) -> Aabb {
        self.objects.iter().fold(
            Aabb {
                min: Vec3::splat(f32::INFINITY),
                max: Vec3::splat(-f32::INFINITY),
            },
            |aabb, obj| obj.get_aabb().surrounding(&aabb),
        )
    }
}

pub struct Bvh {
    left: Option<Box<dyn Hittable>>,
    right: Option<Box<dyn Hittable>>,
    aabb: Aabb,
}

impl Bvh {
    pub fn new(mut objects: Vec<Box<dyn Hittable>>, axis: usize) -> Self {
        if objects.len() == 0 {
            Bvh {
                left: None,
                right: None,
                aabb: Aabb {
                    min: Vec3::zero(),
                    max: Vec3::zero(),
                },
            }
        } else if objects.len() == 1 {
            let left = objects.pop().unwrap();
            let aabb = left.get_aabb();
            Bvh {
                left: Some(left),
                right: None,
                aabb,
            }
        } else if objects.len() == 2 {
            let left = objects.pop().unwrap();
            let right = objects.pop().unwrap();
            let aabb = Aabb::surrounding(&left.get_aabb(), &right.get_aabb());
            Bvh {
                left: Some(left),
                right: Some(right),
                aabb,
            }
        } else {
            objects.sort_by(|a, b|
                a.get_aabb().min[axis].partial_cmp(&b.get_aabb().min[axis]).unwrap());
            let mut left_objects = objects;
            let right_objects = left_objects.split_off(left_objects.len() / 2);
            let left = Box::new(Bvh::new(left_objects, (axis + 1) % 3));
            let right = Box::new(Bvh::new(right_objects, (axis + 1) % 3));
            let aabb = Aabb::surrounding(&left.get_aabb(), &right.get_aabb());
            Bvh {
                left: Some(left),
                right: Some(right),
                aabb,
            }
        }
    }
}

impl Hittable for Bvh {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, hit: &mut Hit) -> bool {
        if !self.aabb.hit(&ray.origin, &ray.dir, t_min, t_max) {
            return false;
        }

        let hit_left = match &self.left {
            Some(obj) => obj.hit(ray, t_min, t_max, hit),
            None => false,
        };

        let hit_right = match &self.right {
            Some(obj) => obj.hit(ray, t_min, if hit_left { hit.t } else { t_max }, hit),
            None => false,
        };

        hit_left || hit_right
    }

    fn get_aabb(&self) -> Aabb {
        self.aabb
    }
}
