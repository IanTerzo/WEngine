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

        self.cube_handle = Some(
            ctx.spawn(
                EntityBuilder::dynamic_body(Transform {
                    position: self.position,
                    rotation: UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.0f32)
                        .into_inner(),
                    scale: vector![1.0, 1.0, 1.0],
                })
                .mesh(cube_mesh)
                .collider_cuboid(vector![1.0, 1.0, 1.0])
                .tag("walkable"),
            ),
        );
    }

    fn on_update(&mut self, delta: f32, ctx: &mut SceneContext) {
        if let Some(cube_handle) = self.cube_handle.clone() {
            let cube_entity = ctx
                .get_entity(cube_handle)
                .unwrap()
                .into_dynamicbody()
                .unwrap();

            let rot_increment = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 1.0 * delta);

            let current_rot =
                UnitQuaternion::from_quaternion(cube_entity.entity.transform.rotation);

            let new_rot = current_rot * rot_increment;

            cube_entity.entity.transform.rotation = new_rot.into_inner();
        }
    }
}
