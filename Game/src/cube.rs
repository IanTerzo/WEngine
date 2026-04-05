use nalgebra::{self, UnitQuaternion, Vector3, vector};
use wengine::{
    entity::{EntityHandle, builder::EntityBuilder},
    scene::{Scene, SceneContext},
    transform::Transform,
};

pub struct Cube {
    position: Vector3<f32>,
    cube_handle: Option<EntityHandle>,
}

impl Cube {
    pub fn new(position: Vector3<f32>) -> Self {
        Self {
            position,
            cube_handle: None,
        }
    }
}

impl Scene for Cube {
    fn on_init(&mut self, ctx: &mut SceneContext) {
        let cube_mesh = ctx.load_obj("assets/cube.obj").unwrap()[0];

        self.cube_handle = ctx
            .spawn(
                EntityBuilder::dynamic_body(Transform {
                    position: self.position,
                    rotation: UnitQuaternion::identity(),
                    scale: vector![1.0, 1.0, 1.0],
                })
                .mesh(cube_mesh)
                .collider_cuboid(vector![1.0, 1.0, 1.0])
                .tag("walkable"),
            )
            .ok()
    }
}
