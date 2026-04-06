use winit::keyboard::PhysicalKey;

use crate::{
    EngineState,
    entity::{
        Entity, EntityHandle,
        builder::EntityBuilder,
        delete::DeleteContext,
        get_entity_from_handle,
        refs::{
            CameraRef, DynamicBodyRef, EmptyRef, EntityRef, KinematicBodyRef, MeshInstanceRef,
            PointLightRef, StaticBodyRef,
        },
        spawn::SpawnContext,
    },
    mesh::{MeshHandle, load_obj},
};

pub enum EngineEvent {
    Key {
        physical_key: PhysicalKey,
        pressed: bool,
    },
    MouseMotion {
        delta_x: f64,
        delta_y: f64,
    },
    MouseButton {
        button: winit::event::MouseButton,
        pressed: bool,
    },
    CollisionEnter {
        entity: EntityHandle,
        other: EntityHandle,
    },
    CollisionExit {
        entity: EntityHandle,
        other: EntityHandle,
    },
}

pub trait Scene {
    fn on_init(&mut self, _ctx: &mut SceneContext) {}
    fn on_update(&mut self, _delta_time: f32, _ctx: &mut SceneContext) {}
    fn on_physics_update(&mut self, _delta_time: f32, _ctx: &mut SceneContext) {}
    fn on_event(&mut self, _event: EngineEvent, _ctx: &mut SceneContext) {}
}

pub struct SceneInstance {
    pub scene: Box<dyn Scene>,
    pub is_active: bool,
}

pub struct SceneContext<'a> {
    engine_state: &'a mut EngineState,
    scenes: &'a mut Vec<SceneInstance>,
}

// User facing API

impl<'a> SceneContext<'a> {
    pub(crate) fn new(state: &'a mut EngineState, scenes: &'a mut Vec<SceneInstance>) -> Self {
        Self {
            engine_state: state,
            scenes,
        }
    }

    pub fn get_entity(&mut self, entity_handle: EntityHandle) -> anyhow::Result<EntityRef<'_>> {
        let entity = get_entity_from_handle(&mut self.engine_state.entities, entity_handle)?;
        match entity {
            Entity::StaticBody(e) => Ok(EntityRef::StaticBody(StaticBodyRef { entity: e })),
            Entity::DynamicBody(e) => Ok(EntityRef::DynamicBody(DynamicBodyRef {
                entity: e,
                physics_world: &mut self.engine_state.physics_world,
            })),
            Entity::KinematicBody(e) => Ok(EntityRef::KinematicBody(KinematicBodyRef {
                entity: e,
                physics_world: &mut self.engine_state.physics_world,
            })),
            Entity::MeshInstance(e) => Ok(EntityRef::MeshInstance(MeshInstanceRef { entity: e })),
            Entity::Camera(e) => Ok(EntityRef::Camera(CameraRef { entity: e })),
            Entity::Empty(e) => Ok(EntityRef::Empty(EmptyRef { entity: e })),
            Entity::PointLight(e) => Ok(EntityRef::PointLight(PointLightRef { entity: e })),
        }
    }

    pub fn spawn(&mut self, entity: impl Into<EntityBuilder>) -> anyhow::Result<EntityHandle> {
        SpawnContext {
            entities: &mut self.engine_state.entities,
            root_entities: &mut self.engine_state.root_entities,
            meshes: &mut self.engine_state.meshes,
            collider_entity_pairs: &mut self.engine_state.collider_entity_pairs,
            physics_world: &mut self.engine_state.physics_world,
            camera: &mut self.engine_state.camera,
            lighting: &mut self.engine_state.lighting,
            queue: &self.engine_state.renderer.queue,
            config: &self.engine_state.renderer.config,
        }
        .spawn(entity)
    }

    pub fn delete(&mut self, entity: EntityHandle) -> anyhow::Result<()> {
        DeleteContext {
            entities: &mut self.engine_state.entities,
            root_entities: &mut self.engine_state.root_entities,
            meshes: &mut self.engine_state.meshes,
            collider_entity_pairs: &mut self.engine_state.collider_entity_pairs,
            physics_world: &mut self.engine_state.physics_world,
            camera: &mut self.engine_state.camera,
            lighting: &mut self.engine_state.lighting,
            queue: &self.engine_state.renderer.queue,
            config: &self.engine_state.renderer.config,
        }
        .delete(entity)
    }

    pub fn get_entities_by_tag(&self, tag: &str) -> Vec<EntityHandle> {
        let mut result = Vec::new();

        for (index, entity) in &self.engine_state.entities {
            let entity_tag = match entity {
                Entity::DynamicBody(e) => &e.tag,
                Entity::StaticBody(e) => &e.tag,
                Entity::KinematicBody(e) => &e.tag,
                Entity::MeshInstance(e) => &e.tag,
                Entity::Camera(e) => &e.tag,
                Entity::Empty(e) => &e.tag,
                Entity::PointLight(e) => &e.tag,
            };

            if *entity_tag == Some(tag.to_string()) {
                result.push(EntityHandle(index));
            }
        }

        result
    }

    pub fn spawn_scene(&mut self, scene: impl Scene + 'static) {
        self.scenes.push(SceneInstance {
            scene: Box::new(scene),
            is_active: false,
        });
    }

    pub fn load_obj(&mut self, path: &str) -> anyhow::Result<Vec<MeshHandle>> {
        if let Some(handles) = self.engine_state.mesh_registry.get(path) {
            return Ok(handles.clone());
        }

        let handles = load_obj(
            &self.engine_state.renderer.device,
            &self.engine_state.renderer.queue,
            &self.engine_state.renderer.texture_bind_group_layout,
            path,
            &mut self.engine_state.meshes,
        )?;

        self.engine_state
            .mesh_registry
            .insert(path.to_string(), handles.clone());

        Ok(handles)
    }

    pub fn grab_cursor(&mut self) {
        self.engine_state.cursor_grabbed = true;
        let _ = self
            .engine_state
            .window
            .set_cursor_grab(winit::window::CursorGrabMode::Confined);
        self.engine_state.window.set_cursor_visible(false);
    }

    pub fn release_cursor(&mut self) {
        self.engine_state.cursor_grabbed = false;
        let _ = self
            .engine_state
            .window
            .set_cursor_grab(winit::window::CursorGrabMode::None);
        self.engine_state.window.set_cursor_visible(true);
    }
}
