use anyhow::{Ok, anyhow};
pub mod builder;
pub mod refs;
pub mod spawn;

use crate::{instance::InstanceHandle, lightning::LightHandle, transform::Transform};

#[derive(Clone, Debug)]
pub struct StaticBody {
    pub tag: Option<String>,
    pub transform: Transform,
    pub children: Vec<Entity>,
    pub instance_handle: Option<InstanceHandle>,
    pub rigid_body_handle: rapier3d::prelude::RigidBodyHandle,
}

#[derive(Clone, Debug)]
pub struct DynamicBody {
    pub tag: Option<String>,
    pub transform: Transform,
    pub children: Vec<Entity>,
    pub instance_handle: Option<InstanceHandle>,
    pub rigid_body_handle: rapier3d::prelude::RigidBodyHandle,
}

#[derive(Clone, Debug)]
pub struct KinematicBody {
    pub tag: Option<String>,
    pub transform: Transform,
    pub children: Vec<Entity>,
    pub instance_handle: Option<InstanceHandle>,
    pub rigid_body_handle: rapier3d::prelude::RigidBodyHandle,
}

#[derive(Clone, Debug)]
pub struct Camera {
    pub tag: Option<String>,
    pub transform: Transform,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

#[derive(Clone, Debug)]
pub struct MeshInstance {
    pub tag: Option<String>,
    pub transform: Transform,
    pub children: Vec<Entity>,
    pub instance_handle: InstanceHandle,
}

#[derive(Clone, Debug)]
pub struct Empty {
    pub tag: Option<String>,
    pub transform: Transform,
    pub children: Vec<Entity>,
}

#[derive(Clone, Debug)]
pub struct PointLight {
    pub tag: Option<String>,
    pub transform: Transform,
    pub children: Vec<Entity>,
    pub color: [f32; 3],
    pub strenght: f32,
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

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EntityHandle {
    root: usize,
    path: Vec<usize>,
}

fn children_mut(entity: &mut Entity) -> Option<&mut Vec<Entity>> {
    match entity {
        Entity::DynamicBody(e) => Some(&mut e.children),
        Entity::StaticBody(e) => Some(&mut e.children),
        Entity::KinematicBody(e) => Some(&mut e.children),
        Entity::MeshInstance(e) => Some(&mut e.children),
        Entity::Empty(e) => Some(&mut e.children),
        Entity::Camera(_) => None, // cameras have no children
        Entity::PointLight(e) => Some(&mut e.children),
    }
}

pub fn get_entity_from_handle<'a>(
    entities: &'a mut Vec<Entity>,
    entity_handle: EntityHandle,
) -> anyhow::Result<&'a mut Entity> {
    let mut entity: &mut Entity = entities
        .get_mut(entity_handle.root)
        .ok_or_else(|| anyhow!("Invalid root index: {}", entity_handle.root))?;

    for &index in &entity_handle.path {
        let children =
            children_mut(entity).ok_or_else(|| anyhow!("Entity in path has no children"))?;
        entity = children
            .get_mut(index)
            .ok_or_else(|| anyhow!("Invalid child index: {}", index))?;
    }
    Ok(entity)
}
