use nalgebra::{UnitQuaternion, Vector3};
use rapier3d::prelude::*;

use crate::{
    EngineState, RigidBodyData,
    model::{Instance, InstanceHandle, MeshHandle, Transform},
};

pub struct PhysicsWorld {
    pub gravity: nalgebra::Vector3<f32>,
    pub integration_parameters: IntegrationParameters,
    pub physics_pipeline: PhysicsPipeline,
    pub island_manager: IslandManager,
    pub broad_phase: DefaultBroadPhase,
    pub narrow_phase: NarrowPhase,
    pub impulse_joint_set: ImpulseJointSet,
    pub multibody_joint_set: MultibodyJointSet,
    pub ccd_solver: CCDSolver,
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            gravity: nalgebra::Vector3::new(0.0, -9.82, 0.0),
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
        }
    }

    pub fn step(&mut self) {
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            &(),
            &(),
        );
    }
}

pub enum ColliderConfig {
    Ball { radius: f32 },
    Capsule { half_height: f32, radius: f32 },
    Cuboid { half_extents: Vector<f32> },
    Cylinder { half_height: f32, radius: f32 },
    Custom(rapier3d::prelude::Collider),
}
pub struct DynamicBody {
    pub mesh_handle: MeshHandle,
    pub transform: Transform,
    pub collider: ColliderConfig,
    pub linear_velocity: Vector3<f32>,
    pub angular_velocity: Vector3<f32>,
    pub mass: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub gravity_scale: f32,
    pub can_sleep: bool,
}

pub struct StaticBody {
    pub mesh_handle: MeshHandle,
    pub transform: Transform,
    pub collider: ColliderConfig,
}

pub struct KinematicBody {
    pub mesh_handle: MeshHandle,
    pub transform: Transform,
    pub collider: ColliderConfig,
    pub linear_velocity: Vector3<f32>,
    pub angular_velocity: Vector3<f32>,
}

// Enum to pass to spawn
pub enum Body {
    Dynamic(DynamicBody),
    Static(StaticBody),
    Kinematic(KinematicBody),
}

impl Body {
    pub fn dynamic(mesh_handle: MeshHandle, transform: Transform) -> DynamicBody {
        DynamicBody::new(mesh_handle, transform)
    }

    pub fn static_body(mesh_handle: MeshHandle, transform: Transform) -> StaticBody {
        StaticBody::new(mesh_handle, transform)
    }

    pub fn kinematic(mesh_handle: MeshHandle, transform: Transform) -> KinematicBody {
        KinematicBody::new(mesh_handle, transform)
    }
}

impl DynamicBody {
    pub fn new(mesh_handle: MeshHandle, transform: Transform) -> Self {
        Self {
            mesh_handle,
            transform,
            collider: ColliderConfig::Ball { radius: 1.0 },
            linear_velocity: Vector3::zeros(),
            angular_velocity: Vector3::zeros(),
            mass: 1.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            gravity_scale: 1.0,
            can_sleep: true,
        }
    }

    pub fn collider_ball(mut self, radius: f32) -> Self {
        self.collider = ColliderConfig::Ball { radius };
        self
    }

    pub fn collider_capsule(mut self, half_height: f32, radius: f32) -> Self {
        self.collider = ColliderConfig::Capsule {
            half_height,
            radius,
        };
        self
    }

    pub fn collider_cuboid(mut self, half_extents: Vector3<f32>) -> Self {
        self.collider = ColliderConfig::Cuboid { half_extents };
        self
    }

    pub fn collider_cylinder(mut self, half_height: f32, radius: f32) -> Self {
        self.collider = ColliderConfig::Cylinder {
            half_height,
            radius,
        };
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

impl StaticBody {
    pub fn new(mesh_handle: MeshHandle, transform: Transform) -> Self {
        Self {
            mesh_handle,
            transform,
            collider: ColliderConfig::Ball { radius: 1.0 },
        }
    }

    pub fn collider_ball(mut self, radius: f32) -> Self {
        self.collider = ColliderConfig::Ball { radius };
        self
    }

    pub fn collider_capsule(mut self, half_height: f32, radius: f32) -> Self {
        self.collider = ColliderConfig::Capsule {
            half_height,
            radius,
        };
        self
    }

    pub fn collider_cuboid(mut self, half_extents: Vector3<f32>) -> Self {
        self.collider = ColliderConfig::Cuboid { half_extents };
        self
    }

    pub fn collider_cylinder(mut self, half_height: f32, radius: f32) -> Self {
        self.collider = ColliderConfig::Cylinder {
            half_height,
            radius,
        };
        self
    }
}

impl KinematicBody {
    pub fn new(mesh_handle: MeshHandle, transform: Transform) -> Self {
        Self {
            mesh_handle,
            transform,
            collider: ColliderConfig::Ball { radius: 1.0 },
            linear_velocity: Vector3::zeros(),
            angular_velocity: Vector3::zeros(),
        }
    }

    pub fn collider_ball(mut self, radius: f32) -> Self {
        self.collider = ColliderConfig::Ball { radius };
        self
    }

    pub fn collider_capsule(mut self, half_height: f32, radius: f32) -> Self {
        self.collider = ColliderConfig::Capsule {
            half_height,
            radius,
        };
        self
    }

    pub fn collider_cuboid(mut self, half_extents: Vector3<f32>) -> Self {
        self.collider = ColliderConfig::Cuboid { half_extents };
        self
    }

    pub fn collider_cylinder(mut self, half_height: f32, radius: f32) -> Self {
        self.collider = ColliderConfig::Cylinder {
            half_height,
            radius,
        };
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

// Convert each type to the Body enum
impl From<DynamicBody> for Body {
    fn from(body: DynamicBody) -> Self {
        Body::Dynamic(body)
    }
}

impl From<StaticBody> for Body {
    fn from(body: StaticBody) -> Self {
        Body::Static(body)
    }
}

impl From<KinematicBody> for Body {
    fn from(body: KinematicBody) -> Self {
        Body::Kinematic(body)
    }
}
