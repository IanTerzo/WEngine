use nalgebra::{self, vector, UnitQuaternion, Vector3};
use rand::random_range;
use winit::keyboard::{KeyCode, PhysicalKey};
use WEngine::{
    model::{MeshHandle, Transform},
    physics::Body,
    CameraInfo, EngineEvent, Game, Runner, Scene,
};

struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

struct MyGame {
    camera: CameraInfo,
    camera_controller: CameraController,
    cube_mesh: Option<MeshHandle>,
}

impl MyGame {
    fn new() -> Self {
        Self {
            camera: CameraInfo::default(),
            camera_controller: CameraController {
                speed: 0.2,
                is_backward_pressed: false,
                is_forward_pressed: false,
                is_left_pressed: false,
                is_right_pressed: false,
            },
            cube_mesh: None,
        }
    }
}

impl Game for MyGame {
    fn on_init(&mut self, scene: &mut Scene) {
        scene.grab_cursor();
        let cube_mesh = scene
            .load_obj("/home/ianterzo/Work/WEngine/res/cube.obj")
            .unwrap()[0];

        scene.spawn(
            Body::static_body(
                cube_mesh,
                Transform {
                    position: vector![0.0, -10.0, 0.0],
                    rotation: UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.0f32)
                        .into_inner(),
                    scale: vector![10.0, 1.0, 10.0],
                },
            )
            .collider_cuboid(vector![10.0, 1.0, 10.0]),
        );

        self.cube_mesh = Some(cube_mesh);
    }

    fn on_update(&mut self, delta: f32, scene: &mut Scene) {
        let forward = self.camera.target - self.camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        // Prevents glitching when the camera gets too close to the
        // center of the scene.
        if self.camera_controller.is_forward_pressed && forward_mag > self.camera_controller.speed {
            self.camera.eye += forward_norm * self.camera_controller.speed;
        }
        if self.camera_controller.is_backward_pressed {
            self.camera.eye -= forward_norm * self.camera_controller.speed;
        }

        let right = forward_norm.cross(&self.camera.up);

        // Redo radius calc in case the forward/backward is pressed.
        let forward = self.camera.target - self.camera.eye;
        let forward_mag = forward.magnitude();

        let speed = self.camera_controller.speed as f32;

        if self.camera_controller.is_right_pressed {
            // Rescale the distance between the target and the eye so
            // that it doesn't change. The eye, therefore, still
            // lies on the circle made by the target and eye.
            self.camera.eye =
                self.camera.target - (forward - right * speed).normalize() * forward_mag
        }
        if self.camera_controller.is_left_pressed {
            self.camera.eye =
                self.camera.target - (forward + right * speed).normalize() * forward_mag;
        }

        scene.update_camera(&self.camera);
    }

    fn on_event(&mut self, event: EngineEvent, scene: &mut Scene) {
        match event {
            EngineEvent::Key {
                physical_key,
                pressed,
            } => match physical_key {
                PhysicalKey::Code(code) => match code {
                    KeyCode::KeyW | KeyCode::ArrowUp => {
                        self.camera_controller.is_forward_pressed = pressed;
                    }
                    KeyCode::KeyA | KeyCode::ArrowLeft => {
                        self.camera_controller.is_left_pressed = pressed;
                    }
                    KeyCode::KeyS | KeyCode::ArrowDown => {
                        self.camera_controller.is_backward_pressed = pressed;
                    }
                    KeyCode::KeyD | KeyCode::ArrowRight => {
                        self.camera_controller.is_right_pressed = pressed;
                    }
                    KeyCode::KeyQ => {
                        if !pressed {
                            return;
                        }

                        if let Some(cube_mesh) = self.cube_mesh {
                            scene.spawn(
                                Body::dynamic(
                                    cube_mesh,
                                    Transform {
                                        position: vector![
                                            random_range(-5..5) as f32,
                                            0.0,
                                            random_range(-5..5) as f32
                                        ],
                                        rotation: UnitQuaternion::from_axis_angle(
                                            &Vector3::y_axis(),
                                            0.0f32,
                                        )
                                        .into_inner(),
                                        scale: vector![1.0, 1.0, 1.0],
                                    },
                                )
                                .collider_cuboid(vector![1.0, 1.0, 1.0]),
                            );
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

fn main() -> anyhow::Result<()> {
    Runner::run(MyGame::new())
}
