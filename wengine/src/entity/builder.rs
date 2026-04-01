use nalgebra::Vector3;

use crate::{mesh::MeshHandle, physics::ColliderConfig, transform::Transform};

pub enum EntityBuilder {
    DynamicBody(DynamicBodyBuilder),
    StaticBody(StaticBodyBuilder),
    KinematicBody(KinematicBodyBuilder),
    MeshInstance(MeshInstanceBuilder),
    Camera(CameraBuilder),
    Empty(EmptyBuilder),
    PointLight(PointLightBuilder),
}

impl EntityBuilder {
    pub fn dynamic_body(transform: Transform) -> DynamicBodyBuilder {
        DynamicBodyBuilder::new(transform)
    }

    pub fn static_body(transform: Transform) -> StaticBodyBuilder {
        StaticBodyBuilder::new(transform)
    }

    pub fn kinematic_body(transform: Transform) -> KinematicBodyBuilder {
        KinematicBodyBuilder::new(transform)
    }

    pub fn mesh_instance(mesh_handle: MeshHandle, transform: Transform) -> MeshInstanceBuilder {
        MeshInstanceBuilder::new(mesh_handle, transform)
    }

    pub fn camera(transform: Transform) -> CameraBuilder {
        CameraBuilder::new(transform)
    }

    pub fn empty(transform: Transform) -> EmptyBuilder {
        EmptyBuilder::new(transform)
    }

    pub fn point_light(transform: Transform) -> PointLightBuilder {
        PointLightBuilder::new(transform)
    }
}

pub struct DynamicBodyBuilder {
    pub tag: Option<String>,
    pub mesh_handle: Option<MeshHandle>,
    pub transform: Transform,
    pub children: Vec<EntityBuilder>,
    pub collider: Option<ColliderConfig>,
    pub linear_velocity: Vector3<f32>,
    pub angular_velocity: Vector3<f32>,
    pub mass: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub gravity_scale: f32,
    pub can_sleep: bool,
}

pub struct StaticBodyBuilder {
    pub tag: Option<String>,
    pub mesh_handle: Option<MeshHandle>,
    pub transform: Transform,
    pub children: Vec<EntityBuilder>,
    pub collider: Option<ColliderConfig>,
}

pub struct KinematicBodyBuilder {
    pub tag: Option<String>,
    pub mesh_handle: Option<MeshHandle>,
    pub transform: Transform,
    pub children: Vec<EntityBuilder>,
    pub collider: Option<ColliderConfig>,
    pub linear_velocity: Vector3<f32>,
    pub angular_velocity: Vector3<f32>,
}

pub struct MeshInstanceBuilder {
    pub tag: Option<String>,
    pub mesh_handle: MeshHandle,
    pub transform: Transform,
    pub children: Vec<EntityBuilder>,
}

pub struct CameraBuilder {
    pub tag: Option<String>,
    pub transform: Transform,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

pub struct EmptyBuilder {
    pub tag: Option<String>,
    pub transform: Transform,
    pub children: Vec<EntityBuilder>,
}

pub struct PointLightBuilder {
    pub tag: Option<String>,
    pub mesh_handle: Option<MeshHandle>,
    pub transform: Transform,
    pub children: Vec<EntityBuilder>,
    pub color: [f32; 3],
    pub strength: f32,
}

impl DynamicBodyBuilder {
    pub fn new(transform: Transform) -> Self {
        Self {
            tag: None,
            mesh_handle: None,
            transform,
            children: vec![],
            collider: None,
            linear_velocity: Vector3::zeros(),
            angular_velocity: Vector3::zeros(),
            mass: 1.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            gravity_scale: 1.0,
            can_sleep: true,
        }
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    pub fn mesh(mut self, mesh: MeshHandle) -> Self {
        self.mesh_handle = Some(mesh);
        self
    }

    pub fn add_child(mut self, child: impl Into<EntityBuilder>) -> Self {
        self.children.push(child.into());
        self
    }

    pub fn collider_ball(mut self, radius: f32) -> Self {
        self.collider = Some(ColliderConfig::Ball { radius });
        self
    }

    pub fn collider_capsule(mut self, half_height: f32, radius: f32) -> Self {
        self.collider = Some(ColliderConfig::Capsule {
            half_height,
            radius,
        });
        self
    }

    pub fn collider_cuboid(mut self, half_extents: Vector3<f32>) -> Self {
        self.collider = Some(ColliderConfig::Cuboid { half_extents });
        self
    }

    pub fn collider_cylinder(mut self, half_height: f32, radius: f32) -> Self {
        self.collider = Some(ColliderConfig::Cylinder {
            half_height,
            radius,
        });
        self
    }

    pub fn linear_velocity(mut self, velocity: Vector3<f32>) -> Self {
        self.linear_velocity = velocity;
        self
    }

    pub fn angular_velocity(mut self, velocity: Vector3<f32>) -> Self {
        self.angular_velocity = velocity;
        self
    }

    pub fn mass(mut self, mass: f32) -> Self {
        self.mass = mass;
        self
    }

    pub fn linear_damping(mut self, damping: f32) -> Self {
        self.linear_damping = damping;
        self
    }

    pub fn angular_damping(mut self, damping: f32) -> Self {
        self.angular_damping = damping;
        self
    }

    pub fn gravity_scale(mut self, scale: f32) -> Self {
        self.gravity_scale = scale;
        self
    }

    pub fn can_sleep(mut self, can_sleep: bool) -> Self {
        self.can_sleep = can_sleep;
        self
    }
}

impl StaticBodyBuilder {
    pub fn new(transform: Transform) -> Self {
        Self {
            tag: None,
            mesh_handle: None,
            transform,
            children: vec![],
            collider: None,
        }
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    pub fn mesh(mut self, mesh: MeshHandle) -> Self {
        self.mesh_handle = Some(mesh);
        self
    }

    pub fn add_child(mut self, child: impl Into<EntityBuilder>) -> Self {
        self.children.push(child.into());
        self
    }

    pub fn collider_ball(mut self, radius: f32) -> Self {
        self.collider = Some(ColliderConfig::Ball { radius });
        self
    }

    pub fn collider_capsule(mut self, half_height: f32, radius: f32) -> Self {
        self.collider = Some(ColliderConfig::Capsule {
            half_height,
            radius,
        });
        self
    }

    pub fn collider_cuboid(mut self, half_extents: Vector3<f32>) -> Self {
        self.collider = Some(ColliderConfig::Cuboid { half_extents });
        self
    }

    pub fn collider_cylinder(mut self, half_height: f32, radius: f32) -> Self {
        self.collider = Some(ColliderConfig::Cylinder {
            half_height,
            radius,
        });
        self
    }
}

impl KinematicBodyBuilder {
    pub fn new(transform: Transform) -> Self {
        Self {
            tag: None,
            mesh_handle: None,
            transform,
            children: vec![],
            collider: None,
            linear_velocity: Vector3::zeros(),
            angular_velocity: Vector3::zeros(),
        }
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    pub fn mesh(mut self, mesh: MeshHandle) -> Self {
        self.mesh_handle = Some(mesh);
        self
    }

    pub fn add_child(mut self, child: impl Into<EntityBuilder>) -> Self {
        self.children.push(child.into());
        self
    }

    pub fn collider_ball(mut self, radius: f32) -> Self {
        self.collider = Some(ColliderConfig::Ball { radius });
        self
    }

    pub fn collider_capsule(mut self, half_height: f32, radius: f32) -> Self {
        self.collider = Some(ColliderConfig::Capsule {
            half_height,
            radius,
        });
        self
    }

    pub fn collider_cuboid(mut self, half_extents: Vector3<f32>) -> Self {
        self.collider = Some(ColliderConfig::Cuboid { half_extents });
        self
    }

    pub fn collider_cylinder(mut self, half_height: f32, radius: f32) -> Self {
        self.collider = Some(ColliderConfig::Cylinder {
            half_height,
            radius,
        });
        self
    }

    pub fn linear_velocity(mut self, velocity: Vector3<f32>) -> Self {
        self.linear_velocity = velocity;
        self
    }

    pub fn angular_velocity(mut self, velocity: Vector3<f32>) -> Self {
        self.angular_velocity = velocity;
        self
    }
}

impl MeshInstanceBuilder {
    pub fn new(mesh_handle: MeshHandle, transform: Transform) -> Self {
        Self {
            tag: None,
            mesh_handle,
            transform,
            children: vec![],
        }
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    pub fn add_child(mut self, child: impl Into<EntityBuilder>) -> Self {
        self.children.push(child.into());
        self
    }
}

impl CameraBuilder {
    pub fn new(transform: Transform) -> Self {
        Self {
            tag: None,
            transform,
            fov: 45.0,
            near: 0.1,
            far: 100.0,
        }
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    pub fn fov(mut self, fov: f32) -> Self {
        self.fov = fov;
        self
    }

    pub fn near(mut self, near: f32) -> Self {
        self.near = near;
        self
    }

    pub fn far(mut self, far: f32) -> Self {
        self.far = far;
        self
    }
}

impl EmptyBuilder {
    pub fn new(transform: Transform) -> Self {
        Self {
            tag: None,
            transform,
            children: vec![],
        }
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    pub fn add_child(mut self, child: impl Into<EntityBuilder>) -> Self {
        self.children.push(child.into());
        self
    }
}

impl PointLightBuilder {
    pub fn new(transform: Transform) -> Self {
        Self {
            tag: None,
            mesh_handle: None,
            transform,
            children: vec![],
            color: [1.0, 1.0, 1.0],
            strength: 1.0,
        }
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    pub fn add_child(mut self, child: impl Into<EntityBuilder>) -> Self {
        self.children.push(child.into());
        self
    }

    pub fn color(mut self, color: [f32; 3]) -> Self {
        self.color = color;
        self
    }

    pub fn strength(mut self, strength: f32) -> Self {
        self.strength = strength;
        self
    }

    pub fn mesh(mut self, mesh: MeshHandle) -> Self {
        self.mesh_handle = Some(mesh);
        self
    }
}

impl From<DynamicBodyBuilder> for EntityBuilder {
    fn from(body: DynamicBodyBuilder) -> Self {
        EntityBuilder::DynamicBody(body)
    }
}

impl From<StaticBodyBuilder> for EntityBuilder {
    fn from(body: StaticBodyBuilder) -> Self {
        EntityBuilder::StaticBody(body)
    }
}

impl From<KinematicBodyBuilder> for EntityBuilder {
    fn from(body: KinematicBodyBuilder) -> Self {
        EntityBuilder::KinematicBody(body)
    }
}

impl From<CameraBuilder> for EntityBuilder {
    fn from(camera: CameraBuilder) -> Self {
        EntityBuilder::Camera(camera)
    }
}

impl From<MeshInstanceBuilder> for EntityBuilder {
    fn from(mesh_instance: MeshInstanceBuilder) -> Self {
        EntityBuilder::MeshInstance(mesh_instance)
    }
}
impl From<EmptyBuilder> for EntityBuilder {
    fn from(empty: EmptyBuilder) -> Self {
        EntityBuilder::Empty(empty)
    }
}

impl From<PointLightBuilder> for EntityBuilder {
    fn from(empty: PointLightBuilder) -> Self {
        EntityBuilder::PointLight(empty)
    }
}
