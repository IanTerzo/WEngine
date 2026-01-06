use nalgebra::Vector3;
use rapier3d::prelude::*;

use crate::entity::{EntityHandle, EntityRef, get_entity_from_handle};

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

// Rigidbody functions    TODO: Handle erros properly, and rework entity system.

pub fn apply_impulse(
    physics_world: &mut PhysicsWorld,
    entities: &mut Vec<EntityRef>,
    entity_handle: EntityHandle,
    vector: Vector3<f32>,
) {
    match get_entity_from_handle(entities, entity_handle) {
        EntityRef::DynamicBody(rigid_body)
        | EntityRef::StaticBody(rigid_body)
        | EntityRef::KinematicBody(rigid_body) => {
            let body = physics_world
                .rigid_body_set
                .get_mut(rigid_body.rigid_body_handle)
                .unwrap();

            body.apply_impulse(vector, true);
        }
        _ => return,
    }
}

pub fn add_force(
    physics_world: &mut PhysicsWorld,
    entities: &mut Vec<EntityRef>,
    entity_handle: EntityHandle,
    vector: Vector3<f32>,
) {
    match get_entity_from_handle(entities, entity_handle) {
        EntityRef::DynamicBody(rigid_body)
        | EntityRef::StaticBody(rigid_body)
        | EntityRef::KinematicBody(rigid_body) => {
            let body = physics_world
                .rigid_body_set
                .get_mut(rigid_body.rigid_body_handle)
                .unwrap();

            body.add_force(vector, true);
        }
        _ => return,
    }
}

pub fn get_linvel(
    physics_world: &mut PhysicsWorld,
    entities: &mut Vec<EntityRef>,
    entity_handle: EntityHandle,
) -> Vector3<f32> {
    match get_entity_from_handle(entities, entity_handle) {
        EntityRef::DynamicBody(rigid_body)
        | EntityRef::StaticBody(rigid_body)
        | EntityRef::KinematicBody(rigid_body) => {
            let body = physics_world
                .rigid_body_set
                .get_mut(rigid_body.rigid_body_handle)
                .unwrap();

            *body.linvel()
        }
        _ => return Vector3::zeros(),
    }
}

pub fn set_linvel(
    physics_world: &mut PhysicsWorld,
    entities: &mut Vec<EntityRef>,
    entity_handle: EntityHandle,
    vector: Vector3<f32>,
) {
    match get_entity_from_handle(entities, entity_handle) {
        EntityRef::DynamicBody(rigid_body)
        | EntityRef::StaticBody(rigid_body)
        | EntityRef::KinematicBody(rigid_body) => {
            let body = physics_world
                .rigid_body_set
                .get_mut(rigid_body.rigid_body_handle)
                .unwrap();

            body.set_linvel(vector, true);
        }
        _ => return,
    }
}

pub fn get_angvel(
    physics_world: &mut PhysicsWorld,
    entities: &mut Vec<EntityRef>,
    entity_handle: EntityHandle,
) -> Vector3<f32> {
    match get_entity_from_handle(entities, entity_handle) {
        EntityRef::DynamicBody(rigid_body)
        | EntityRef::StaticBody(rigid_body)
        | EntityRef::KinematicBody(rigid_body) => {
            let body = physics_world
                .rigid_body_set
                .get_mut(rigid_body.rigid_body_handle)
                .unwrap();

            *body.angvel()
        }
        _ => return Vector3::zeros(),
    }
}

pub fn set_angvel(
    physics_world: &mut PhysicsWorld,
    entities: &mut Vec<EntityRef>,
    entity_handle: EntityHandle,
    vector: Vector3<f32>,
) {
    match get_entity_from_handle(entities, entity_handle) {
        EntityRef::DynamicBody(rigid_body)
        | EntityRef::StaticBody(rigid_body)
        | EntityRef::KinematicBody(rigid_body) => {
            let body = physics_world
                .rigid_body_set
                .get_mut(rigid_body.rigid_body_handle)
                .unwrap();

            body.set_angvel(vector, true);
        }
        _ => return,
    }
}

pub fn set_enabled_rotations(
    physics_world: &mut PhysicsWorld,
    entities: &mut Vec<EntityRef>,
    entity_handle: EntityHandle,
    enable_x: bool,
    enable_y: bool,
    enable_z: bool,
) {
    match get_entity_from_handle(entities, entity_handle) {
        EntityRef::DynamicBody(rigid_body)
        | EntityRef::StaticBody(rigid_body)
        | EntityRef::KinematicBody(rigid_body) => {
            if let Some(body) = physics_world
                .rigid_body_set
                .get_mut(rigid_body.rigid_body_handle)
            {
                body.set_enabled_rotations(enable_x, enable_y, enable_z, true);
            }
        }
        _ => return,
    }
}
