use nalgebra::{Isometry, Perspective3, Translation3, Vector3};

use crate::{
    camera::CameraState,
    entity::Entity,
    instance::{InstanceHandle, InstanceType},
    lightning::LightingState,
    mesh::MeshData,
    physics::PhysicsWorld,
    renderer::OPENGL_TO_WGPU_MATRIX,
    transform::Transform,
};

pub struct UpdateContext<'a> {
    pub meshes: &'a mut Vec<MeshData>,
    pub camera: &'a mut CameraState,
    pub lighting: &'a mut LightingState,
    pub physics_world: &'a PhysicsWorld,
    pub queue: &'a wgpu::Queue,
    pub config: &'a wgpu::SurfaceConfiguration,
}

impl<'a> UpdateContext<'a> {
    fn update_instance(&mut self, handle: InstanceHandle, transform: Transform) {
        match handle.instance_type {
            InstanceType::Standard => {
                if let Some(mesh_data) = self.meshes.get_mut(handle.mesh.0) {
                    if let Some(instance) =
                        mesh_data.standard_instances.get_mut(handle.instance_index)
                    {
                        instance.transform = transform;
                    }
                }
            }
            InstanceType::Light => {
                if let Some(mesh_data) = self.meshes.get_mut(handle.mesh.0) {
                    if let Some(instance) = mesh_data.light_instances.get_mut(handle.instance_index)
                    {
                        instance.transform = transform;
                    }
                }
            }
        }
    }

    pub fn update_entity(&mut self, entity_ref: &Entity, parent_transform: Transform) {
        // We want to update all non rigidbody children with the physics of the parent rigidbody.
        match &entity_ref {
            Entity::DynamicBody(entity) => {
                let rigid_body_calc = self
                    .physics_world
                    .rigid_body_set
                    .get(entity.rigid_body_handle)
                    .unwrap();

                let iso = rigid_body_calc.position();
                let position: Vector3<f32> = iso.translation.vector;
                let rotation = iso.rotation;

                // We apply the physics of the parent rigidbody on all children
                for child in &entity.children {
                    self.update_entity(
                        child,
                        Transform {
                            position,
                            rotation,
                            scale: entity.transform.scale,
                        },
                    );
                }

                if let Some(instance_handle) = entity.instance_handle {
                    self.update_instance(
                        instance_handle,
                        Transform {
                            position,
                            rotation,
                            scale: entity.transform.scale,
                        },
                    );
                }
            }
            Entity::StaticBody(entity) => {
                let rigid_body_calc = self
                    .physics_world
                    .rigid_body_set
                    .get(entity.rigid_body_handle)
                    .unwrap();

                let iso = rigid_body_calc.position();
                let position: Vector3<f32> = iso.translation.vector;
                let rotation = iso.rotation;

                for child in &entity.children {
                    self.update_entity(
                        child,
                        Transform {
                            position,
                            rotation,
                            scale: entity.transform.scale,
                        },
                    );
                }

                if let Some(instance_handle) = entity.instance_handle {
                    self.update_instance(
                        instance_handle,
                        Transform {
                            position,
                            rotation,
                            scale: entity.transform.scale,
                        },
                    );
                }
            }
            Entity::KinematicBody(entity) => {
                let rigid_body_calc = self
                    .physics_world
                    .rigid_body_set
                    .get(entity.rigid_body_handle)
                    .unwrap();

                let iso = rigid_body_calc.position();
                let position: Vector3<f32> = iso.translation.vector;
                let rotation = iso.rotation;

                for child in &entity.children {
                    self.update_entity(
                        child,
                        Transform {
                            position,
                            rotation,
                            scale: entity.transform.scale,
                        },
                    );
                }

                if let Some(instance_handle) = entity.instance_handle {
                    self.update_instance(
                        instance_handle,
                        Transform {
                            position,
                            rotation,
                            scale: entity.transform.scale,
                        },
                    );
                }
            }
            Entity::MeshInstance(entity) => {
                let updated_transform = parent_transform.transform(&entity.transform);

                for child in &entity.children {
                    self.update_entity(child, updated_transform);
                }

                self.update_instance(entity.instance_handle, updated_transform);
            }
            Entity::Camera(entity) => {
                let rotated_offset = parent_transform
                    .rotation
                    .transform_vector(&entity.transform.position);

                let camera_position = parent_transform.position + rotated_offset;

                let iso = Isometry::from_parts(
                    Translation3::from(camera_position), // Use the rotated position
                    parent_transform.rotation * entity.transform.rotation,
                );
                let view = iso.inverse().to_homogeneous();
                let aspect = self.config.width as f32 / self.config.height as f32;
                let proj =
                    Perspective3::new(aspect, entity.fov.to_radians(), entity.near, entity.far)
                        .to_homogeneous();
                self.camera
                    .update_view_proj(&self.queue, OPENGL_TO_WGPU_MATRIX * proj * view);
            }
            Entity::Empty(entity) => {
                for child in &entity.children {
                    self.update_entity(child, parent_transform.transform(&entity.transform));
                }
            }
            Entity::PointLight(entity) => {
                let updated_transform = parent_transform.transform(&entity.transform);

                self.lighting.lights[entity.light_handle.0].position =
                    updated_transform.position.into();
                self.lighting.lights[entity.light_handle.0].color = entity.color;
                self.lighting.lights[entity.light_handle.0].strength = entity.strength;

                for child in &entity.children {
                    self.update_entity(child, updated_transform);
                }

                if let Some(instance_handle) = entity.instance_handle {
                    self.update_instance(instance_handle, updated_transform);
                }
            }
        }
    }
}
