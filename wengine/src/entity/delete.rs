use rapier3d::prelude::ColliderHandle;
use slab::Slab;
use std::collections::HashMap;

use crate::{
    camera::CameraState,
    entity::{
        Camera, DynamicBody, Empty, Entity, EntityHandle, KinematicBody, MeshInstance, PointLight,
        StaticBody,
    },
    instance::InstanceHandle,
    lightning::LightingState,
    mesh::MeshData,
    physics::PhysicsWorld,
};

pub struct DeleteContext<'a> {
    pub entities: &'a mut Slab<Entity>,
    pub root_entities: &'a mut Vec<EntityHandle>,
    pub meshes: &'a mut Vec<MeshData>,
    pub collider_entity_pairs: &'a mut HashMap<ColliderHandle, EntityHandle>,
    pub physics_world: &'a mut PhysicsWorld,
    pub camera: &'a mut CameraState,
    pub lighting: &'a mut LightingState,
    pub queue: &'a wgpu::Queue,
    pub config: &'a wgpu::SurfaceConfiguration,
}

impl<'a> DeleteContext<'a> {
    pub fn delete(&mut self, handle: EntityHandle) -> anyhow::Result<()> {
        if !self.entities.contains(handle.0) {
            return Ok(()); // Already deleted
        }

        // Remove from root_entities if present
        self.root_entities.retain(|h| h.0 != handle.0);

        let entity = self.entities.remove(handle.0);

        match entity {
            Entity::DynamicBody(entity) => self.delete_dynamic(entity),
            Entity::StaticBody(entity) => self.delete_static(entity),
            Entity::KinematicBody(entity) => self.delete_kinematic(entity),
            Entity::MeshInstance(entity) => self.delete_mesh_instance(entity),
            Entity::Camera(entity) => self.delete_camera(entity),
            Entity::Empty(entity) => self.delete_empty(entity),
            Entity::PointLight(entity) => self.delete_point_light(entity),
        }
    }

    fn delete_children(&mut self, children: &Vec<EntityHandle>) -> anyhow::Result<()> {
        for child in children {
            self.delete(*child)?
        }
        Ok(())
    }

    fn remove_standard_instance(&mut self, handle: InstanceHandle) -> anyhow::Result<()> {
        let mesh_data = self.meshes.get_mut(handle.mesh.0).unwrap();

        mesh_data.standard_instances.remove(handle.instance_index);

        Ok(())
    }

    fn delete_dynamic(&mut self, entity: DynamicBody) -> anyhow::Result<()> {
        self.delete_children(&entity.children)?;

        if let Some(instance_handle) = entity.instance_handle {
            self.remove_standard_instance(instance_handle)?;
        }

        self.physics_world.rigid_body_set.remove(
            entity.rigid_body_handle,
            &mut self.physics_world.island_manager,
            &mut self.physics_world.collider_set,
            &mut self.physics_world.impulse_joint_set,
            &mut self.physics_world.multibody_joint_set,
            true,
        );

        Ok(())
    }

    fn delete_static(&mut self, entity: StaticBody) -> anyhow::Result<()> {
        self.delete_children(&entity.children)?;

        if let Some(instance_handle) = entity.instance_handle {
            self.remove_standard_instance(instance_handle)?;
        }

        self.physics_world.rigid_body_set.remove(
            entity.rigid_body_handle,
            &mut self.physics_world.island_manager,
            &mut self.physics_world.collider_set,
            &mut self.physics_world.impulse_joint_set,
            &mut self.physics_world.multibody_joint_set,
            true,
        );

        Ok(())
    }

    fn delete_kinematic(&mut self, entity: KinematicBody) -> anyhow::Result<()> {
        self.delete_children(&entity.children)?;

        if let Some(instance_handle) = entity.instance_handle {
            self.remove_standard_instance(instance_handle)?;
        }

        self.physics_world.rigid_body_set.remove(
            entity.rigid_body_handle,
            &mut self.physics_world.island_manager,
            &mut self.physics_world.collider_set,
            &mut self.physics_world.impulse_joint_set,
            &mut self.physics_world.multibody_joint_set,
            true,
        );

        Ok(())
    }

    fn delete_mesh_instance(&mut self, entity: MeshInstance) -> anyhow::Result<()> {
        self.delete_children(&entity.children)?;

        self.remove_standard_instance(entity.instance_handle)?;

        Ok(())
    }

    fn delete_camera(&mut self, _entity: Camera) -> anyhow::Result<()> {
        // We just keep the camera in it's last position

        Ok(())
    }

    fn delete_empty(&mut self, entity: Empty) -> anyhow::Result<()> {
        self.delete_children(&entity.children)?;

        Ok(())
    }

    fn delete_point_light(&mut self, entity: PointLight) -> anyhow::Result<()> {
        self.delete_children(&entity.children)?;

        self.lighting.lights.remove(entity.light_handle.0);

        if let Some(instance_handle) = entity.instance_handle {
            let mesh_data = self.meshes.get_mut(instance_handle.mesh.0).unwrap();

            mesh_data
                .light_instances
                .remove(instance_handle.instance_index);
        }

        Ok(())
    }
}
