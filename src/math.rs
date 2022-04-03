use glam::Vec3;

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

    pub fn random_in_disk(&mut self) -> Vec3 {
        loop {
            let p = Vec3::new(self.random(-1.0, 1.0), self.random(-1.0, 1.0), 0.0);
            if p.length_squared() < 1.0 {
                return p;
            }
        }
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

pub fn rgb_to_u32(col: Vec3) -> u32 {
    ((255.999 * col.x()) as u32) << 16
        | ((255.999 * col.y()) as u32) << 8
        | ((255.999 * col.z()) as u32)
}

#[derive(Copy, Clone)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn hit(&self, ray_from: &Vec3, ray_dir: &Vec3, mut t_min: f32, mut t_max: f32) -> bool {
        for axis in 0..3 {
            let inv_d = 1.0 / ray_dir[axis];
            let mut t0 = (self.min[axis] - ray_from[axis]) * inv_d;
            let mut t1 = (self.max[axis] - ray_from[axis]) * inv_d;
            if inv_d < 0.0 {
                std::mem::swap(&mut t0, &mut t1);
            }
            t_min = if t0 > t_min { t0 } else { t_min };
            t_max = if t1 < t_max { t1 } else { t_max };
            if t_max <= t_min {
                return false;
            }
        }
        true
    }

    pub fn surrounding(&self, aabb: &Aabb) -> Aabb {
        Aabb {
            min: Vec3::min(self.min, aabb.min),
            max: Vec3::max(self.max, aabb.max),
        }
    }
}
