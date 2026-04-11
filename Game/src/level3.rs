use crate::{handle_scene_switch, player::Player};
use nalgebra::{UnitQuaternion, vector};
use wengine::{
    entity::EntityHandle,
    entity::builder::EntityBuilder,
    scene::{EngineEvent, Scene, SceneContext},
    transform::Transform,
};

const NUM_LIGHTS: usize = 6;

pub struct Level3 {
    player: Player,
    light_handles: Vec<EntityHandle>,
    time: f32,
}

impl Level3 {
    pub fn new() -> Self {
        Self {
            player: Player::new(),
            light_handles: Vec::new(),
            time: 0.0,
        }
    }
}

impl Scene for Level3 {
    fn on_init(&mut self, ctx: &mut SceneContext) {
        let cube_mesh = ctx.load_obj("assets/cube.obj").unwrap()[0];

        ctx.spawn(
            EntityBuilder::static_body(Transform {
                position: vector![0.0, -8.0, 0.0],
                rotation: UnitQuaternion::identity(),
                scale: vector![80.0, 1.0, 80.0],
            })
            .collider_cuboid(vector![80.0, 1.0, 80.0])
            .mesh(cube_mesh)
            .tag("walkable"),
        )
        .unwrap();

        let colors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ];

        for i in 0..NUM_LIGHTS {
            let handle = ctx
                .spawn(
                    EntityBuilder::point_light(Transform {
                        position: vector![0.0, 0.0, 0.0],
                        rotation: UnitQuaternion::identity(),
                        scale: vector![0.3, 0.3, 0.3],
                    })
                    .color(colors[i])
                    .strength(1.5)
                    .mesh(cube_mesh),
                )
                .unwrap();
            self.light_handles.push(handle);
        }

        self.player.init(ctx);
    }

    fn on_update(&mut self, delta: f32, ctx: &mut SceneContext) {
        self.time += delta;

        for (i, handle) in self.light_handles.iter().enumerate() {
            if let Ok(entity_ref) = ctx.get_entity(*handle) {
                let light = entity_ref.into_pointlight().unwrap();
                let phase = (i as f32 / NUM_LIGHTS as f32) * std::f32::consts::TAU;
                let t = self.time + phase;

                let radius = 20.0 + 8.0 * (t * 0.3).sin();
                let height = 6.0 * (t * 0.7 + phase).sin();

                light.entity.transform.position =
                    vector![t.sin() * radius, height, t.cos() * radius];
            }
        }
    }

    fn on_physics_update(&mut self, delta: f32, ctx: &mut SceneContext) {
        self.player.on_physics_update(delta, ctx);
    }

    fn on_event(&mut self, event: EngineEvent, ctx: &mut SceneContext) {
        self.player.on_event(&event, ctx);
        handle_scene_switch(&event, ctx);
    }
}
