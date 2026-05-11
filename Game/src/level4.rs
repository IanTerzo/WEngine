use crate::{handle_scene_switch, player::Player};
use nalgebra::{UnitQuaternion, Vector3, vector};
use wengine::{
    entity::EntityHandle,
    entity::builder::EntityBuilder,
    scene::{EngineEvent, Scene, SceneContext},
    transform::Transform,
};

pub struct Level4 {
    player: Player,
    og_cube: Option<EntityHandle>,
    time: f32,
}

impl Level4 {
    pub fn new() -> Self {
        Self {
            player: Player::new(),
            og_cube: None,
            time: 0.0,
        }
    }
}

impl Scene for Level4 {
    fn on_init(&mut self, ctx: &mut SceneContext) {
        let cube_mesh = ctx.load_obj("assets/cube.obj").unwrap()[0];
        let og_cube_mesh = ctx.load_obj("assets/og/untitled.obj").unwrap()[0];

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

        let handle = ctx
            .spawn(EntityBuilder::mesh_instance(
                og_cube_mesh,
                Transform {
                    position: vector![0.0, -4.0, 0.0],
                    rotation: UnitQuaternion::identity(),
                    scale: vector![2.0, 2.0, 2.0],
                },
            ))
            .unwrap();

        self.og_cube = Some(handle);

        ctx.spawn(
            EntityBuilder::point_light(Transform {
                position: vector![0.0, 5.0, 18.0],
                rotation: UnitQuaternion::identity(),
                scale: vector![0.6, 0.6, 0.6],
            })
            .strength(2.0)
            .mesh(cube_mesh),
        )
        .unwrap();

        self.player.init(ctx);
    }

    fn on_update(&mut self, delta: f32, ctx: &mut SceneContext) {
        self.time += delta;

        if let Some(handle) = self.og_cube {
            if let Ok(entity_ref) = ctx.get_entity(handle) {
                let body = entity_ref.into_mesh_instance().unwrap();
                body.entity.transform.rotation =
                    UnitQuaternion::from_axis_angle(&Vector3::y_axis(), self.time * 1.8);
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
