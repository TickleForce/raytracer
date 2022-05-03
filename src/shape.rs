use crate::{camera::*, material::*, math::*};
use glam::{vec3, Mat4, Vec3};
use std::collections::HashMap;
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

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub p: Vec3,
    pub n: Vec3,
}

impl Vertex {
    pub fn new(p: Vec3, n: Vec3) -> Self {
        Vertex { p, n }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Triangle {
    pub v1: Vertex,
    pub v2: Vertex,
    pub v3: Vertex,
    pub material_index: usize,
}

impl Triangle {
    pub fn new(v1: Vertex, v2: Vertex, v3: Vertex, material_index: usize) -> Self {
        Triangle {
            v1,
            v2,
            v3,
            material_index,
        }
    }

    pub fn get_aabb(&self) -> Aabb {
        Aabb::new(&self.v1.p, &self.v1.p)
            .surrounding_point(&self.v2.p)
            .surrounding_point(&self.v3.p)
    }
}

#[derive(Copy, Clone, Debug)]
enum MeshBvhNode {
    Triangle(Triangle),
    Node(usize, usize, Aabb),
    None,
}

pub struct Mesh {
    triangles: Vec<Triangle>,
    transform: Mat4,
    inverse_transform: Mat4,
    materials: Vec<Arc<dyn Material>>,
    bvh_nodes: Vec<MeshBvhNode>,
    aabb: Aabb,
}

fn compute_aabb(triangles: &Vec<Triangle>) -> Aabb {
    let mut aabb = triangles.iter().fold(Aabb::infinity(), |aabb, tri| {
        aabb.surrounding_point(&tri.v1.p)
            .surrounding_point(&tri.v2.p)
            .surrounding_point(&tri.v3.p)
    });

    // prevents zero thickness
    aabb.max[0] += 0.0001;
    aabb.max[1] += 0.0001;
    aabb.max[2] += 0.0001;
    aabb.min[0] -= 0.0001;
    aabb.min[1] -= 0.0001;
    aabb.min[2] -= 0.0001;
    aabb
}

impl Mesh {
    pub fn new(
        triangles: Vec<Triangle>,
        materials: Vec<Arc<dyn Material>>,
        transform: Mat4,
    ) -> Self {
        let aabb = triangles.iter().fold(Aabb::infinity(), |aabb, tri| {
            aabb.surrounding_point(&transform.transform_point3(tri.v1.p))
                .surrounding_point(&transform.transform_point3(tri.v2.p))
                .surrounding_point(&transform.transform_point3(tri.v3.p))
        });

        let mut mesh = Mesh {
            triangles,
            materials,
            transform,
            inverse_transform: transform.inverse(),
            bvh_nodes: Vec::new(),
            aabb,
        };
        mesh.build_bvh(mesh.triangles.clone(), 0);
        mesh
    }

    fn make_triangle_node(&mut self, tri: Triangle) -> usize {
        self.bvh_nodes.push(MeshBvhNode::Triangle(tri));
        self.bvh_nodes.len() - 1
    }

    fn build_bvh(&mut self, mut tris: Vec<Triangle>, axis: usize) -> usize {
        if tris.len() == 0 {
            //println!("None");
            self.bvh_nodes.push(MeshBvhNode::None);
            self.bvh_nodes.len() - 1
        } else if tris.len() == 1 {
            //println!("Triangle");
            self.make_triangle_node(tris.pop().unwrap())
        } else if tris.len() == 2 {
            //println!("BVH_Node -> tri, tri");
            let mut aabb = compute_aabb(&tris);
            let left = self.make_triangle_node(tris.pop().unwrap());
            let right = self.make_triangle_node(tris.pop().unwrap());
            self.bvh_nodes.push(MeshBvhNode::Node(left, right, aabb));
            self.bvh_nodes.len() - 1
        } else {
            //println!("BVH_Node -> BVH_Node");
            tris.sort_by(|a, b| {
                a.get_aabb().min[axis]
                    .partial_cmp(&b.get_aabb().min[axis])
                    .unwrap()
            });
            let mut aabb = compute_aabb(&tris);
            let mut left_tris = tris;
            let right_tris = left_tris.split_off(left_tris.len() / 2);
            let left = self.build_bvh(left_tris, (axis + 1) % 3);
            let right = self.build_bvh(right_tris, (axis + 1) % 3);
            self.bvh_nodes.push(MeshBvhNode::Node(left, right, aabb));
            self.bvh_nodes.len() - 1
        }
    }

    fn bvh_hit(&self, node_index: usize, ray: &Ray, t_min: f32, t_max: f32, hit: &mut Hit) -> bool {
        let node = self.bvh_nodes[node_index];
        if let MeshBvhNode::Triangle(triangle) = node {
            let (is_hit, is_backface, t, u, v) =
                ray_triangle_intersection(&ray, &triangle.v1.p, &triangle.v2.p, &triangle.v3.p);
            if is_hit && t >= t_min && t < t_max {
                let mut hit_normal = triangle.v1.n;
                if is_backface {
                    hit_normal = hit_normal * -1.0;
                }
                hit.t = t;
                hit.p = self.transform.transform_point3(ray.at(t));
                hit.set_face_normal(
                    &ray,
                    &self.transform.transform_vector3(hit_normal).normalize(),
                );
                hit.set_material(self.materials[triangle.material_index].clone());
                true
            } else {
                false
            }
        } else if let MeshBvhNode::Node(left, right, aabb) = node {
            if !aabb.hit(&ray.origin, &ray.dir, t_min, t_max) {
                return false;
            }
            let hit_left = self.bvh_hit(left, ray, t_min, t_max, hit);
            let hit_right =
                self.bvh_hit(right, ray, t_min, if hit_left { hit.t } else { t_max }, hit);
            hit_left || hit_right
        } else {
            false
        }
    }

    pub fn plane(w: f32, h: f32, material: Arc<dyn Material>, transform: Mat4) -> Self {
        Mesh::new(
            vec![
                Triangle::new(
                    Vertex::new(vec3(-w, -h, 0.0), vec3(0.0, 0.0, 1.0)),
                    Vertex::new(vec3(-w, h, 0.0), vec3(0.0, 0.0, 1.0)),
                    Vertex::new(vec3(w, -h, 0.0), vec3(0.0, 0.0, 1.0)),
                    0,
                ),
                Triangle::new(
                    Vertex::new(vec3(w, -h, 0.0), vec3(0.0, 0.0, 1.0)),
                    Vertex::new(vec3(-w, h, 0.0), vec3(0.0, 0.0, 1.0)),
                    Vertex::new(vec3(w, h, 0.0), vec3(0.0, 0.0, 1.0)),
                    0,
                ),
            ],
            vec![material],
            transform,
        )
    }

    pub fn from_file(
        filename: &str,
        material_map: HashMap<String, Arc<dyn Material>>,
        transform: Mat4,
    ) -> Self {
        let (obj_models, obj_materials) = tobj::load_obj(
            filename,
            &tobj::LoadOptions {
                single_index: true,
                triangulate: true,
                ignore_points: true,
                ignore_lines: true,
                ..Default::default()
            },
        )
        .expect("Failed to load OBJ file");
        let obj_materials = obj_materials.expect("Failed to load materials");

        let mut materials: Vec<Arc<dyn Material>> = Vec::new();
        let mut material_index_map = HashMap::new();
        for (material_name, material) in &material_map {
            material_index_map.insert(material_name, materials.len());
            materials.push(material.clone());
        }

        let mut triangles: Vec<Triangle> = Vec::new();
        for (i, m) in obj_models.iter().enumerate() {
            let mesh = &m.mesh;
            let mat = &obj_materials[mesh.material_id.unwrap_or(0)];

            let material_index = if let Some(material_index) = material_index_map.get(&mat.name) {
                *material_index
            } else {
                0
            };
            let vertices = mesh
                .indices
                .iter()
                .map(|index| {
                    Vertex::new(
                        vec3(
                            mesh.positions[(3 * index) as usize],
                            mesh.positions[(3 * index + 1) as usize],
                            mesh.positions[(3 * index + 2) as usize],
                        ),
                        vec3(
                            mesh.normals[(3 * index) as usize],
                            mesh.normals[(3 * index + 1) as usize],
                            mesh.normals[(3 * index + 2) as usize],
                        ),
                    )
                })
                .collect::<Vec<_>>();
            for verts in vertices.chunks(3) {
                triangles.push(Triangle::new(verts[0], verts[1], verts[2], material_index));
            }
        }

        Mesh::new(triangles, materials, transform)
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

    let inv_det = 1.0 / det;

    let tvec = ray.origin - *v0;
    let u = tvec.dot(pvec) * inv_det;
    if u < 0.0 || u > 1.0 {
        return (false, false, 0.0, 0.0, 0.0);
    }

    let qvec = tvec.cross(v0v1);
    let v = ray.dir.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return (false, false, 0.0, 0.0, 0.0);
    }

    let t = v0v2.dot(qvec) * inv_det;
    return (true, is_backface, t, u, v);
}

impl Hittable for Mesh {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, hit: &mut Hit) -> bool {
        let ray = ray.transform(&self.inverse_transform);
        self.bvh_hit(self.bvh_nodes.len() - 1, &ray, t_min, t_max, hit)

        /*
        let mut any_hit = false;
        let mut min_t = t_max;
        let mut hit_normal = Vec3::zero();
        let mut hit_material_index = 0;
        for triangle in self.triangles.iter() {
            let (is_hit, is_backface, t, u, v) =
                ray_triangle_intersection(&ray, &triangle.v1.p, &triangle.v2.p, &triangle.v3.p);
            if is_hit && t >= t_min && t < min_t {
                any_hit = true;
                min_t = t;
                hit_normal = triangle.v1.n;
                hit_material_index = triangle.material_index;
                if is_backface {
                    hit_normal = hit_normal * -1.0;
                }
            }
        }

        if !any_hit {
            return false;
        }

        hit.t = min_t;
        hit.p = self.transform.transform_point3(ray.at(hit.t));
        hit.set_face_normal(
            &ray,
            &self.transform.transform_vector3(hit_normal).normalize(),
        );
        hit.set_material(self.materials[hit_material_index].clone());
        true
        */
    }

    fn get_aabb(&self) -> Aabb {
        self.aabb
    }
}
