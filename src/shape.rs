use crate::camera::*;
use crate::material::*;
use crate::math::*;
use glam::vec3;
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
            objects.sort_by(|a, b| {
                a.get_aabb().min[axis]
                    .partial_cmp(&b.get_aabb().min[axis])
                    .unwrap()
            });
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

pub struct Vertex {
    pub p: Vec3,
    pub n: Vec3,
}

pub struct Mesh {
    verts: Vec<Vertex>,
    pub material: Arc<dyn Material>,
}

impl Mesh {
    pub fn new(verts: Vec<Vertex>, material: Arc<dyn Material>) -> Self {
        Mesh { verts, material }
    }

    pub fn plane(s: f32, z: f32, material: Arc<dyn Material>) -> Self {
        Mesh {
            verts: vec![
                Vertex {
                    p: vec3(-s, -s, z),
                    n: vec3(0.0, 0.0, 1.0),
                },
                Vertex {
                    p: vec3(-s, s, z),
                    n: vec3(0.0, 0.0, 1.0),
                },
                Vertex {
                    p: vec3(s, -s, z),
                    n: vec3(0.0, 0.0, 1.0),
                },
                Vertex {
                    p: vec3(s, -s, z),
                    n: vec3(0.0, 0.0, 1.0),
                },
                Vertex {
                    p: vec3(-s, s, z),
                    n: vec3(0.0, 0.0, 1.0),
                },
                Vertex {
                    p: vec3(s, s, z),
                    n: vec3(0.0, 0.0, 1.0),
                },
            ],
            material,
        }
    }
}

// -> (hit, backface, t, u, v)
// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
fn ray_triangle_intersection(
    ray: &Ray,
    v0: &Vec3,
    v1: &Vec3,
    v2: &Vec3,
) -> (bool, bool, f32, f32, f32) {
    let v0v1 = *v1 - *v0;
    let v0v2 = *v2 - *v0;
    let pvec = ray.dir.cross(v0v2);
    let det = v0v1.dot(pvec);

    let culling = false;
    if culling {
        if det < std::f32::EPSILON {
            return (false, false, 0.0, 0.0, 0.0);
        }
    } else {
        if f32::abs(det) < f32::EPSILON {
            return (false, false, 0.0, 0.0, 0.0);
        }
    }
    let is_backface = det < std::f32::EPSILON;

    let invDet = 1.0 / det;

    let tvec = ray.origin - *v0;
    let u = tvec.dot(pvec) * invDet;
    if u < 0.0 || u > 1.0 {
        return (false, false, 0.0, 0.0, 0.0);
    }

    let qvec = tvec.cross(v0v1);
    let v = ray.dir.dot(qvec) * invDet;
    if v < 0.0 || u + v > 1.0 {
        return (false, false, 0.0, 0.0, 0.0);
    }

    let t = v0v2.dot(qvec) * invDet;
    return (true, is_backface, t, u, v);
}

impl Hittable for Mesh {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, hit: &mut Hit) -> bool {
        let mut any_hit = false;
        let mut min_t = t_max;
        let mut hit_normal = Vec3::zero();
        for verts in self.verts.chunks(3) {
            let (is_hit, is_backface, t, u, v) =
                ray_triangle_intersection(ray, &verts[0].p, &verts[1].p, &verts[2].p);
            if is_hit && t >= t_min && t < min_t {
                any_hit = true;
                min_t = t;
                hit_normal = verts[0].n;
                if is_backface {
                    hit_normal = hit_normal * -1.0;
                }
            }
        }

        if !any_hit {
            return false;
        }

        hit.t = min_t;
        hit.p = ray.at(hit.t);
        hit.set_face_normal(ray, &hit_normal);
        hit.set_material(self.material.clone());

        true
    }

    fn get_aabb(&self) -> Aabb {
        self.verts.iter().fold(
            Aabb {
                min: Vec3::splat(f32::INFINITY),
                max: Vec3::splat(-f32::INFINITY),
            },
            |aabb, v| aabb.surrounding_point(&v.p),
        )
    }
}
