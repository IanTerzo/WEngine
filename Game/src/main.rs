use nalgebra::{self, UnitQuaternion, Vector3, vector};
use rand::random_range;
use wengine::{EngineEvent, Runner, Scene, SceneContext, Transform, entity::EntityBuilder};
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

        let cube_mesh = ctx.load_obj("../res/cube.obj").unwrap()[0];

        ctx.spawn(
            EntityBuilder::static_body(Transform {
                position: vector![0.0, -30.0, 0.0],
                rotation: UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.0f32.to_radians())
                    .into_inner(),
                scale: vector![40.0, 1.0, 40.0],
            })
            .collider_cuboid(vector![40.0, 1.0, 40.0])
            .mesh(cube_mesh)
            .tag("walkable"),
        );

        ctx.spawn(
            EntityBuilder::mesh_instance(
                cube_mesh,
                Transform {
                    position: vector![0.0, -25.0, 0.0],
                    rotation: UnitQuaternion::from_axis_angle(
                        &Vector3::y_axis(),
                        0.0f32.to_radians(),
                    )
                    .into_inner(),
                    scale: vector![1.0, 1.0, 1.0],
                },
            )
            .add_child(EntityBuilder::point_light(Transform {
                position: vector![0.0, 0.0, 0.0],
                rotation: UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.0f32.to_radians())
                    .into_inner(),
                scale: vector![0.0, 0.0, 0.0],
            })),
        );

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
