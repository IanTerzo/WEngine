use cgmath::Rotation3;
use winit::keyboard::{KeyCode, PhysicalKey};
use WEngine::{
    model::{InstanceHandle, Transform},
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
    cube_instance: Option<InstanceHandle>,
    cube_rotation: f32,
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
            cube_instance: None,
            cube_rotation: 0.0,
        }
    }
}

impl Game for MyGame {
    fn on_init(&mut self, scene: &mut Scene) {
        let model = scene
            .load_obj("/home/ianterzo/Work/WEngine/res/cube.obj")
            .unwrap();

        let handle = scene.instantiate(
            model[0],
            Transform {
                position: cgmath::vec3(0.0, 0.0, 0.0),
                rotation: cgmath::Quaternion::from_angle_y(cgmath::Deg(0.0)),
                scale: cgmath::vec3(1.0, 1.0, 1.0),
            },
        );

        self.cube_instance = Some(handle);

        // Load a teapot for reference

        let model = scene
            .load_obj("/home/ianterzo/Work/WEngine/res/teapot.obj")
            .unwrap();

        let _ = scene.instantiate(
            model[0],
            Transform {
                position: cgmath::vec3(0.0, 0.0, -4.0),
                rotation: cgmath::Quaternion::from_angle_y(cgmath::Deg(0.0)),
                scale: cgmath::vec3(1.0, 1.0, 1.0),
            },
        );
    }

    fn on_update(&mut self, delta: f32, scene: &mut Scene) {
        use cgmath::InnerSpace;

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

        let right = forward_norm.cross(self.camera.up);

        // Redo radius calc in case the forward/backward is pressed.
        let forward = self.camera.target - self.camera.eye;
        let forward_mag = forward.magnitude();

        if self.camera_controller.is_right_pressed {
            // Rescale the distance between the target and the eye so
            // that it doesn't change. The eye, therefore, still
            // lies on the circle made by the target and eye.
            self.camera.eye = self.camera.target
                - (forward + right * self.camera_controller.speed).normalize() * forward_mag;
        }
        if self.camera_controller.is_left_pressed {
            self.camera.eye = self.camera.target
                - (forward - right * self.camera_controller.speed).normalize() * forward_mag;
        }

        scene.update_camera(&self.camera);

        // Rotate the cube
        if let Some(handle) = self.cube_instance {
            self.cube_rotation += delta * 80.0;
            scene.update_instance(
                handle,
                Transform {
                    position: cgmath::vec3(0.0, 0.0, 0.0),
                    rotation: cgmath::Quaternion::from_angle_y(cgmath::Deg(self.cube_rotation)),
                    scale: cgmath::vec3(1.0, 1.0, 1.0),
                },
            );
        }
    }

    fn on_event(&mut self, event: EngineEvent, _scene: &mut Scene) {
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
