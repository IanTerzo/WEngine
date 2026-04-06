use crate::{
    camera::CameraState,
    entity::{Entity, EntityHandle, update::UpdateContext},
    lightning::LightingState,
    mesh::{MeshData, MeshHandle},
    physics::PhysicsWorld,
    renderer::Renderer,
    transform::Transform,
};
use rapier3d::prelude::ColliderHandle;
use slab::Slab;
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
    mesh_registry: HashMap<String, Vec<MeshHandle>>,
    entities: Slab<Entity>,
    root_entities: Vec<EntityHandle>,
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
            mesh_registry: HashMap::new(),
            entities: Slab::new(),
            root_entities: vec![],
            collider_entity_pairs: HashMap::new(),
            active_collisions: HashSet::new(),
            cursor_grabbed: false,
        })
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
            .filter(|(handle1, handle2)| {
                // Don't count it as a removed collision if the deleted collision is caused by a deleted entity
                self.entities.get(handle1.0).is_some() && self.entities.get(handle2.0).is_some()
            })
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

        let mut update_context = UpdateContext {
            entities: &self.entities,
            meshes: &mut self.meshes,
            camera: &mut self.camera,
            lighting: &mut self.lighting,
            physics_world: &self.physics_world,
            queue: &self.renderer.queue,
            config: &self.renderer.config,
        };

        for handle in &self.root_entities {
            // If the entity doesn't have a parent, meaning its a root entity,
            // go trough each of its children and apply the parent transform according to the entity type.
            let entity = &self.entities[handle.0];
            update_context.update_entity(entity, Transform::zero());
        }

        // Pass collisions back to App so that the right events can be fired

        self.collect_collisions()
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        // Flush meshes and lights

        self.renderer.flush_meshes(&self.meshes);
        self.lighting.flush(&self.renderer.queue);

        // Render

        self.renderer.render(
            &self.meshes,
            &self.camera.bind_group,
            &self.lighting.bind_group,
        )
    }
}
