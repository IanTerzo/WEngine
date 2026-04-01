use crate::{
    camera::CameraState,
    entity::{
        Entity, EntityHandle,
        builder::EntityBuilder,
        get_entity_from_handle,
        refs::{
            CameraRef, DynamicBodyRef, EmptyRef, EntityRef, KinematicBodyRef, MeshInstanceRef,
            PointLightRef, StaticBodyRef,
        },
        spawn::spawn,
    },
    instance::{InstanceHandle, InstanceRaw, InstanceType},
    lightning::LightingState,
    mesh::{MeshData, MeshHandle, load_obj},
    physics::PhysicsWorld,
    renderer::{OPENGL_TO_WGPU_MATRIX, Renderer},
    transform::Transform,
};
use nalgebra::{self, Isometry, Perspective3, Translation3, UnitQuaternion, Vector3};
use rapier3d::prelude::ColliderHandle;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use winit::window::Window;

pub mod app;
pub mod camera;
pub mod entity;
pub mod instance;
pub mod lightning;
pub mod mesh;
pub mod physics;
pub mod renderer;
pub mod scene;
pub mod texture;
pub mod transform;

pub struct EngineState {
    pub renderer: Renderer,
    pub camera: CameraState,
    pub lighting: LightingState,
    pub physics_world: PhysicsWorld,
    window: Arc<Window>,
    meshes: Vec<MeshData>,
    entities: Vec<Entity>,
    collider_entity_pairs: HashMap<ColliderHandle, EntityHandle>,
    active_collisions: HashSet<(EntityHandle, EntityHandle)>,
    cursor_grabbed: bool,
}

impl EngineState {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<EngineState> {
        let renderer = Renderer::new(window.clone()).await?;

        let camera = CameraState::new(&renderer.device, &renderer.camera_bind_group_layout);

        let lighting = LightingState::new(&renderer.device, &renderer.light_bind_group_layout);

        let physics_world = PhysicsWorld::new();

        Ok(Self {
            renderer,
            camera,
            lighting,
            physics_world,
            window,
            meshes: vec![],
            entities: vec![],
            collider_entity_pairs: HashMap::new(),
            active_collisions: HashSet::new(),
            cursor_grabbed: false,
        })
    }

    pub fn load_obj(&mut self, path: &str) -> anyhow::Result<Vec<MeshHandle>> {
        load_obj(
            &self.renderer.device,
            &self.renderer.queue,
            &self.renderer.texture_bind_group_layout,
            path,
            &mut self.meshes,
        )
    }

    pub fn spawn(&mut self, entity: impl Into<EntityBuilder>) -> EntityHandle {
        spawn(
            &mut self.entities,
            &mut self.meshes,
            &mut self.collider_entity_pairs,
            &mut self.physics_world,
            &self.renderer.queue,
            &mut self.camera.uniform,
            &self.camera.buffer,
            &self.renderer.config,
            &mut self.lighting.lights,
            &self.lighting.buffer,
            entity,
        )
    }

    pub fn get_entity<'a>(
        &'a mut self,
        entity_handle: EntityHandle,
    ) -> anyhow::Result<EntityRef<'a>> {
        let entity = get_entity_from_handle(&mut self.entities, entity_handle)?;
        match entity {
            Entity::StaticBody(e) => Ok(EntityRef::StaticBody(StaticBodyRef { entity: e })),
            Entity::DynamicBody(e) => Ok(EntityRef::DynamicBody(DynamicBodyRef {
                entity: e,
                physics_world: &mut self.physics_world,
            })),
            Entity::KinematicBody(e) => Ok(EntityRef::KinematicBody(KinematicBodyRef {
                entity: e,
                physics_world: &mut self.physics_world,
            })),
            Entity::MeshInstance(e) => Ok(EntityRef::MeshInstance(MeshInstanceRef { entity: e })),
            Entity::Camera(e) => Ok(EntityRef::Camera(CameraRef { entity: e })),
            Entity::Empty(e) => Ok(EntityRef::Empty(EmptyRef { entity: e })),
            Entity::PointLight(e) => Ok(EntityRef::PointLight(PointLightRef { entity: e })),
        }
    }

    pub fn grab_cursor(&mut self) {
        self.cursor_grabbed = true;
        let _ = self
            .window
            .set_cursor_grab(winit::window::CursorGrabMode::Confined);
        self.window.set_cursor_visible(false);
    }

    pub fn release_cursor(&mut self) {
        self.cursor_grabbed = false;
        let _ = self
            .window
            .set_cursor_grab(winit::window::CursorGrabMode::None);
        self.window.set_cursor_visible(true);
    }

    fn update_instance(&mut self, handle: InstanceHandle, transform: Transform) {
        match handle.instance_type {
            InstanceType::Standard => {
                if let Some(mesh_data) = self.meshes.get_mut(handle.mesh.0) {
                    if let Some(instance) =
                        mesh_data.standard_instances.get_mut(handle.instance_index)
                    {
                        instance.transform = transform;

                        let instance_raw = instance.to_raw();
                        let offset = handle.instance_index * std::mem::size_of::<InstanceRaw>();

                        self.renderer.queue.write_buffer(
                            &mesh_data.standard_instance_buffer,
                            offset as wgpu::BufferAddress,
                            bytemuck::cast_slice(&[instance_raw]),
                        );
                    }
                }
            }
            InstanceType::Light => {
                if let Some(mesh_data) = self.meshes.get_mut(handle.mesh.0) {
                    if let Some(instance) = mesh_data.light_instances.get_mut(handle.instance_index)
                    {
                        instance.transform = transform;

                        let instance_raw = instance.to_raw();
                        let offset = handle.instance_index * std::mem::size_of::<InstanceRaw>();

                        self.renderer.queue.write_buffer(
                            &mesh_data.light_instance_buffer,
                            offset as wgpu::BufferAddress,
                            bytemuck::cast_slice(&[instance_raw]),
                        );
                    }
                }
            }
        }
    }

    fn update_entity(&mut self, entity_ref: &Entity, parent_transform: Transform) {
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
                let rotation = iso.rotation.into_inner();

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
                let rotation = iso.rotation.into_inner();

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
                let rotation = iso.rotation.into_inner();

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
                let rotated_offset = UnitQuaternion::from_quaternion(parent_transform.rotation)
                    .transform_vector(&entity.transform.position);

                let camera_position = parent_transform.position + rotated_offset;

                let iso = Isometry::from_parts(
                    Translation3::from(camera_position), // Use the rotated position
                    UnitQuaternion::from_quaternion(parent_transform.rotation)
                        * UnitQuaternion::from_quaternion(entity.transform.rotation),
                );
                let view = iso.inverse().to_homogeneous();
                let aspect = self.renderer.config.width as f32 / self.renderer.config.height as f32;
                let proj =
                    Perspective3::new(aspect, entity.fov.to_radians(), entity.near, entity.far)
                        .to_homogeneous();
                self.camera
                    .update_view_proj(&self.renderer.queue, OPENGL_TO_WGPU_MATRIX * proj * view);
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
                self.lighting.lights[entity.light_handle.0].strength = entity.strenght;

                for child in &entity.children {
                    self.update_entity(child, updated_transform);
                }

                if let Some(instance_handle) = entity.instance_handle {
                    self.update_instance(instance_handle, updated_transform);
                }
            }
        }
    }

    fn collect_collisions(
        &mut self,
    ) -> (
        Vec<(EntityHandle, EntityHandle)>,
        Vec<(EntityHandle, EntityHandle)>,
    ) {
        let current_collisions: HashSet<(EntityHandle, EntityHandle)> = self
            .physics_world
            .narrow_phase
            .contact_pairs()
            .map(|contact_pair| {
                let handle1 = self
                    .collider_entity_pairs
                    .get(&contact_pair.collider1)
                    .unwrap()
                    .clone();
                let handle2 = self
                    .collider_entity_pairs
                    .get(&contact_pair.collider2)
                    .unwrap()
                    .clone();

                // Ensure that it's the same no matter what collider is 1 or 2.
                if handle1 <= handle2 {
                    (handle1, handle2)
                } else {
                    (handle2, handle1)
                }
            })
            .collect();

        let new_collisions: Vec<_> = current_collisions
            .difference(&self.active_collisions)
            .cloned()
            .collect();

        let removed_collisions: Vec<_> = self
            .active_collisions
            .difference(&current_collisions)
            .cloned()
            .collect();

        self.active_collisions = current_collisions;

        (new_collisions, removed_collisions)
    }

    pub fn update(
        &mut self,
    ) -> (
        Vec<(EntityHandle, EntityHandle)>,
        Vec<(EntityHandle, EntityHandle)>,
    ) {
        // Step the physics world

        self.physics_world.step();

        // Update entities

        let entities = std::mem::take(&mut self.entities);
        for entity in &entities {
            self.update_entity(entity, Transform::zero());
        }
        self.entities = entities;

        self.lighting.flush(&self.renderer.queue);

        // Collisions

        self.collect_collisions()
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();
        self.renderer.render(
            &self.meshes,
            &self.camera.bind_group,
            &self.lighting.bind_group,
        )
    }
}
