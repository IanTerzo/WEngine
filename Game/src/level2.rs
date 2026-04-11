use crate::{handle_scene_switch, player::Player};
use nalgebra::{UnitQuaternion, vector};
use wengine::{
    entity::EntityHandle,
    entity::builder::EntityBuilder,
    scene::{EngineEvent, Scene, SceneContext},
    transform::Transform,
};

pub struct Level2 {
    player: Player,
    light_handle: Option<EntityHandle>,
    time: f32,
}

impl Level2 {
    pub fn new() -> Self {
        Self {
            player: Player::new(),
            light_handle: None,
            time: 0.0,
        }
    }
}

impl Scene for Level2 {
    fn on_init(&mut self, ctx: &mut SceneContext) {
        let cube_mesh = ctx.load_obj("assets/cube.obj").unwrap()[0];
        let blue_cube_mesh = ctx.load_obj("assets/blue_cube.obj").unwrap()[0];
        let logo = ctx.load_obj("assets/logo.obj").unwrap()[0];

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

        ctx.spawn(EntityBuilder::mesh_instance(
            logo,
            Transform {
                position: vector![-12.0, -2.0, 0.0],
                rotation: UnitQuaternion::identity(),
                scale: vector![0.1, 0.1, 0.1],
            },
        ))
        .unwrap();

        self.light_handle = ctx
            .spawn(
                EntityBuilder::point_light(Transform {
                    position: vector![0.0, -2.0, 0.0],
                    rotation: UnitQuaternion::identity(),
                    scale: vector![0.6, 0.6, 0.6],
                })
                .color([0.0, 0.0, 1.0])
                .strength(1.0)
                .mesh(blue_cube_mesh),
            )
            .ok();

        self.player.init(ctx);
    }

    fn on_update(&mut self, delta: f32, ctx: &mut SceneContext) {
        self.time += delta;

        if let Some(handle) = self.light_handle {
            if let Ok(entity_ref) = ctx.get_entity(handle) {
                let light = entity_ref.into_pointlight().unwrap();
                let radius = 12.0;
                light.entity.transform.position =
                    vector![self.time.sin() * radius, -2.0, self.time.cos() * radius];
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
