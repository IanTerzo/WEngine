use winit::keyboard::PhysicalKey;

use crate::{
    EngineState,
    entity::{EntityHandle, builder::EntityBuilder, refs::EntityRef},
    mesh::MeshHandle,
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

pub struct SceneInstance {
    pub scene: Box<dyn Scene>,
    pub is_active: bool,
}

pub struct SceneContext<'a> {
    core: &'a mut EngineState,
    scenes: &'a mut Vec<SceneInstance>,
}

pub trait Scene {
    fn on_init(&mut self, _ctx: &mut SceneContext) {}
    fn on_update(&mut self, _delta_time: f32, _ctx: &mut SceneContext) {}
    fn on_event(&mut self, _event: EngineEvent, _ctx: &mut SceneContext) {}
}

// User facing abstraction of EngineState

impl<'a> SceneContext<'a> {
    pub(crate) fn new(state: &'a mut EngineState, scenes: &'a mut Vec<SceneInstance>) -> Self {
        Self {
            core: state,
            scenes,
        }
    }

    pub fn get_entity(&mut self, entity_handle: EntityHandle) -> anyhow::Result<EntityRef<'_>> {
        self.core.get_entity(entity_handle)
    }

    pub fn spawn(&mut self, entity: impl Into<EntityBuilder>) -> EntityHandle {
        self.core.spawn(entity)
    }

    pub fn spawn_scene(&mut self, scene: impl Scene + 'static) {
        self.scenes.push(SceneInstance {
            scene: Box::new(scene),
            is_active: false,
        });
    }

    pub fn load_obj(&mut self, path: &str) -> anyhow::Result<Vec<MeshHandle>> {
        self.core.load_obj(path)
    }

    pub fn grab_cursor(&mut self) {
        self.core.grab_cursor();
    }

    pub fn release_cursor(&mut self) {
        self.core.release_cursor();
    }
}
