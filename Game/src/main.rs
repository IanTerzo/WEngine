use nalgebra::{self, UnitQuaternion, vector};
use rand::random_range;
use wengine::{
    app::Runner,
    entity::builder::EntityBuilder,
    scene::{EngineEvent, Scene, SceneContext},
    transform::Transform,
};
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::{cube::Cube, player::Player};

mod cube;
mod player;

struct Main {}

impl Main {
    fn new() -> Self {
        Self {}
    }
}

impl Scene for Main {
    fn on_init(&mut self, ctx: &mut SceneContext) {
        // World

        let cube_mesh = ctx.load_obj("assets/cube.obj").unwrap()[0];
        let blue_cube_mesh = ctx.load_obj("assets/blue_cube.obj").unwrap()[0];

        ctx.spawn(
            EntityBuilder::static_body(Transform {
                position: vector![0.0, -30.0, 0.0],
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
                position: vector![20.0, -25.5, 0.0],
                rotation: UnitQuaternion::identity(),
                scale: vector![0.6, 0.6, 0.6],
            })
            .mesh(cube_mesh),
        )
        .unwrap();

        ctx.spawn(
            EntityBuilder::point_light(Transform {
                position: vector![-20.0, -25.5, 0.0],
                rotation: UnitQuaternion::identity(),
                scale: vector![0.6, 0.6, 0.6],
            })
            .color([0.0, 0.0, 1.0])
            .mesh(blue_cube_mesh),
        )
        .unwrap();

        // Player

        ctx.spawn_scene(Player::new());
    }

    fn on_event(&mut self, event: EngineEvent, ctx: &mut SceneContext) {
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

                        ctx.spawn_scene(Cube::new(vector![
                            random_range(-5..5) as f32,
                            0.0,
                            random_range(-5..5) as f32
                        ]));
                    }
                    _ => {}
                },
                _ => {}
            },
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    Runner::new(Main::new())
        .window_width(1280)
        .window_height(720)
        .title("First person controller")
        .run()
}
