use crate::math::*;
use crate::camera::*;
use glam::Vec3;

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
    pub albedo: Vec3,
    pub normal: bool,
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
        /*
        let target = hit.p + series.random_in_hemisphere(&hit.normal);
        let d = (target - hit.p).normalize();
        */
        *ray_out = Ray {
            origin: hit.p,
            dir: d.normalize(),
        };
        let surface_color = if self.normal {
            (hit.normal + Vec3::new(1.0, 1.0, 1.0)) * 0.5
        } else {
            self.albedo
        };
        *attenuation = surface_color;
        true
    }
}

pub struct MetalMaterial {
    pub albedo: Vec3,
    pub fuzz: f32,
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
    pub albedo: Vec3,
    pub ior: f32,
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
