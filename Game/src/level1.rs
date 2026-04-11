use nalgebra::{UnitQuaternion, vector};
use rand::random_range;
use wengine::{
    entity::builder::EntityBuilder,
    scene::{EngineEvent, Scene, SceneContext},
    transform::Transform,
};
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::{handle_scene_switch, player::Player};

pub struct Level1 {
    player: Player,
}

impl Level1 {
    pub fn new() -> Self {
        Self {
            player: Player::new(),
        }
    }
}

impl Scene for Level1 {
    fn on_init(&mut self, ctx: &mut SceneContext) {
        let cube_mesh = ctx.load_obj("assets/cube.obj").unwrap()[0];

        ctx.spawn(
            EntityBuilder::static_body(Transform {
                position: vector![0.0, -8.0, 0.0],
                rotation: UnitQuaternion::identity(),
                scale: vector![40.0, 1.0, 40.0],
            })
            .collider_cuboid(vector![40.0, 1.0, 40.0])
            .mesh(cube_mesh)
            .tag("walkable"),
        )
        .unwrap();

        ctx.spawn(
            EntityBuilder::point_light(Transform {
                position: vector![0.0, 10.0, 0.0],
                rotation: UnitQuaternion::identity(),
                scale: vector![0.6, 0.6, 0.6],
            })
            .mesh(cube_mesh),
        )
        .unwrap();

        self.player.init(ctx);
    }

    fn on_physics_update(&mut self, delta: f32, ctx: &mut SceneContext) {
        self.player.on_physics_update(delta, ctx);
    }

    fn on_event(&mut self, event: EngineEvent, ctx: &mut SceneContext) {
        self.player.on_event(&event, ctx);
        handle_scene_switch(&event, ctx);

        match event {
            EngineEvent::Key {
                physical_key,
                pressed,
            } => match physical_key {
                PhysicalKey::Code(code) => match code {
                    KeyCode::KeyQ => {
                        if !pressed {
                            return;
                        }
                        let cube_mesh = ctx.load_obj("assets/cube.obj").unwrap()[0];
                        ctx.spawn(
                            EntityBuilder::dynamic_body(Transform {
                                position: vector![
                                    random_range(-5..5) as f32,
                                    20.0,
                                    random_range(-5..5) as f32
                                ],
                                rotation: UnitQuaternion::identity(),
                                scale: vector![1.0, 1.0, 1.0],
                            })
                            .mesh(cube_mesh)
                            .collider_cuboid(vector![1.0, 1.0, 1.0])
                            .tag("cube"),
                        )
                        .ok();
                    }
                    KeyCode::KeyX => {
                        if !pressed {
                            return;
                        }
                        for entity_handle in ctx.get_entities_by_tag("cube") {
                            ctx.delete(entity_handle).unwrap();
                        }
                    }
                    _ => {}
                },
                _ => {}
            },
            _ => {}
        }
    }
}
