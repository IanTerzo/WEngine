use WEngine::{EngineEvent, Runner, Scene, SceneContext, Transform, entity::EntityBuilder};
use nalgebra::{self, UnitQuaternion, Vector3, vector};
use rand::random_range;
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
        // Ground

        let cube_mesh = ctx.load_obj("../res/cube.obj").unwrap()[0];

        ctx.spawn(
            EntityBuilder::static_body(Transform {
                position: vector![0.0, -30.0, 0.0],
                rotation: UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.0f32.to_radians())
                    .into_inner(),
                scale: vector![10.0, 1.0, 10.0],
            })
            .collider_cuboid(vector![10.0, 1.0, 10.0])
            .mesh(cube_mesh)
            .tag("walkable"),
        );

        // Player

        ctx.instantiate_scene(Player::new());
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

                        ctx.instantiate_scene(Cube::new(vector![
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
