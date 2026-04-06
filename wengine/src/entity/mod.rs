use anyhow::{Ok, anyhow};
use slab::Slab;
pub mod builder;
pub mod delete;
pub mod refs;
pub mod spawn;
pub mod update;

use crate::{instance::InstanceHandle, lightning::LightHandle, transform::Transform};

#[derive(Clone, Debug)]
pub struct StaticBody {
    pub tag: Option<String>,
    pub transform: Transform,
    pub children: Vec<EntityHandle>,
    pub parent: Option<EntityHandle>,
    pub instance_handle: Option<InstanceHandle>,
    pub rigid_body_handle: rapier3d::prelude::RigidBodyHandle,
}

#[derive(Clone, Debug)]
pub struct DynamicBody {
    pub tag: Option<String>,
    pub transform: Transform,
    pub children: Vec<EntityHandle>,
    pub parent: Option<EntityHandle>,
    pub instance_handle: Option<InstanceHandle>,
    pub rigid_body_handle: rapier3d::prelude::RigidBodyHandle,
}

#[derive(Clone, Debug)]
pub struct KinematicBody {
    pub tag: Option<String>,
    pub transform: Transform,
    pub children: Vec<EntityHandle>,
    pub parent: Option<EntityHandle>,
    pub instance_handle: Option<InstanceHandle>,
    pub rigid_body_handle: rapier3d::prelude::RigidBodyHandle,
}

#[derive(Clone, Debug)]
pub struct Camera {
    pub tag: Option<String>,
    pub transform: Transform,
    pub parent: Option<EntityHandle>,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

#[derive(Clone, Debug)]
pub struct MeshInstance {
    pub tag: Option<String>,
    pub transform: Transform,
    pub children: Vec<EntityHandle>,
    pub parent: Option<EntityHandle>,
    pub instance_handle: InstanceHandle,
}

#[derive(Clone, Debug)]
pub struct Empty {
    pub tag: Option<String>,
    pub transform: Transform,
    pub children: Vec<EntityHandle>,
    pub parent: Option<EntityHandle>,
}

#[derive(Clone, Debug)]
pub struct PointLight {
    pub tag: Option<String>,
    pub transform: Transform,
    pub children: Vec<EntityHandle>,
    pub parent: Option<EntityHandle>,
    pub color: [f32; 3],
    pub strength: f32,
    pub instance_handle: Option<InstanceHandle>,
    pub light_handle: LightHandle,
}

#[derive(Clone, Debug)]
pub enum Entity {
    DynamicBody(DynamicBody),
    StaticBody(StaticBody),
    KinematicBody(KinematicBody),
    Camera(Camera),
    MeshInstance(MeshInstance),
    Empty(Empty),
    PointLight(PointLight),
}

impl Entity {
    pub fn set_parent(&mut self, handle: EntityHandle) {
        match self {
            Entity::DynamicBody(e) => e.parent = Some(handle),
            Entity::StaticBody(e) => e.parent = Some(handle),
            Entity::KinematicBody(e) => e.parent = Some(handle),
            Entity::MeshInstance(e) => e.parent = Some(handle),
            Entity::Camera(e) => e.parent = Some(handle),
            Entity::Empty(e) => e.parent = Some(handle),
            Entity::PointLight(e) => e.parent = Some(handle),
        }
    }

    pub fn get_parent(&self) -> Option<EntityHandle> {
        match self {
            Entity::DynamicBody(e) => e.parent,
            Entity::StaticBody(e) => e.parent,
            Entity::KinematicBody(e) => e.parent,
            Entity::MeshInstance(e) => e.parent,
            Entity::Camera(e) => e.parent,
            Entity::Empty(e) => e.parent,
            Entity::PointLight(e) => e.parent,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EntityHandle(pub usize);

pub fn get_entity_from_handle<'a>(
    entities: &'a mut Slab<Entity>,
    entity_handle: EntityHandle,
) -> anyhow::Result<&'a mut Entity> {
    let entity: &mut Entity = entities
        .get_mut(entity_handle.0)
        .ok_or_else(|| anyhow!("Invalid handle with value: {}", entity_handle.0))?;

    Ok(entity)
}
