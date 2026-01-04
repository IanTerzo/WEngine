use nalgebra::{self, vector, UnitQuaternion, Vector3};
use rand::random_range;
use winit::keyboard::{KeyCode, PhysicalKey};
use WEngine::{
    model::{MeshHandle, Transform},
    physics::Entity,
    CameraState, EngineEvent, EntityHandle, EntityState, Game, Runner, Scene,
};

struct CameraController {
    speed: f32,
    sensitivity: f32,
    yaw: f32,   // Rotation around Y axis
    pitch: f32, // Rotation around X axis
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl CameraController {
    fn new() -> Self {
        Self {
            speed: 0.2,
            sensitivity: 0.002,
            yaw: 0.0,
            pitch: 0.0,
            is_backward_pressed: false,
            is_forward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    fn process_mouse(&mut self, delta_x: f64, delta_y: f64) {
        self.yaw -= delta_x as f32 * self.sensitivity;
        self.pitch -= delta_y as f32 * self.sensitivity;

        // Clamp pitch to prevent camera flipping
        self.pitch = self
            .pitch
            .clamp(-89.0f32.to_radians(), 89.0f32.to_radians());
    }

    fn get_rotation(&self) -> nalgebra::Quaternion<f32> {
        let yaw_quat = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), self.yaw);
        let pitch_quat = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), self.pitch);
        (yaw_quat * pitch_quat).into_inner()
    }
}

struct MyGame {
    camera_controller: CameraController,
    cube_mesh: Option<MeshHandle>,
    player_entity: Option<EntityHandle>,
    cursor_grabbed: bool,
}

impl MyGame {
    fn new() -> Self {
        Self {
            camera_controller: CameraController::new(),
            cube_mesh: None,
            player_entity: None,
            cursor_grabbed: false,
        }
    }
}

impl Game for MyGame {
    fn on_init(&mut self, scene: &mut Scene) {
        let cube_mesh = scene
            .load_obj("/home/ianterzo/Work/WEngine/res/cube.obj")
            .unwrap()[0];

        // Ground
        scene.spawn(
            Entity::static_body(
                cube_mesh,
                Transform {
                    position: vector![0.0, -30.0, 0.0],
                    rotation: UnitQuaternion::from_axis_angle(
                        &Vector3::y_axis(),
                        0.0f32.to_radians(),
                    )
                    .into_inner(),
                    scale: vector![10.0, 1.0, 10.0],
                },
            )
            .collider_cuboid(vector![10.0, 1.0, 10.0]),
        );

        // Player with camera
        let player = scene.spawn(
            Entity::dynamic(
                cube_mesh,
                Transform {
                    position: vector![0.0, 0.0, 0.0],
                    rotation: UnitQuaternion::from_axis_angle(
                        &Vector3::y_axis(),
                        0.0f32.to_radians(),
                    )
                    .into_inner(),
                    scale: vector![1.0, 1.0, 1.0],
                },
            )
            .add_child(Entity::camera(Transform {
                position: vector![0.0, 2.0, 0.0],
                rotation: UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.0f32.to_radians())
                    .into_inner(),
                scale: vector![1.0, 1.0, 1.0],
            })),
        );

        self.player_entity = Some(player);
        self.cube_mesh = Some(cube_mesh);

        // Grab cursor for mouse look
        scene.grab_cursor();
        self.cursor_grabbed = true;
    }

    fn on_update(&mut self, delta: f32, scene: &mut Scene) {
        // Update camera rotation based on mouse input
        if let Some(player_handle) = &self.player_entity {
            if let EntityState::Camera(camera_state) =
                &mut scene.get_entity(player_handle.clone()).children[0].state
            {
                *camera_state = CameraState {
                    transform: Transform {
                        position: camera_state.transform.position,
                        rotation: self.camera_controller.get_rotation(),
                        scale: camera_state.transform.scale,
                    },
                    fov: camera_state.fov,
                    near: camera_state.near,
                    far: camera_state.far,
                };
            }
        }
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
                                Entity::dynamic(
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
                    KeyCode::Escape => {
                        if !pressed {
                            return;
                        }

                        // Toggle cursor grab
                        if self.cursor_grabbed {
                            scene.release_cursor();
                            self.cursor_grabbed = false;
                        } else {
                            scene.grab_cursor();
                            self.cursor_grabbed = true;
                        }
                    }
                    _ => {}
                },
                _ => {}
            },
            EngineEvent::MouseMotion { delta_x, delta_y } => {
                if self.cursor_grabbed {
                    self.camera_controller.process_mouse(delta_x, delta_y);
                }
            }
            EngineEvent::MouseButton { button, pressed } => {
                // Optional: grab cursor on mouse click
                if pressed && !self.cursor_grabbed {
                    scene.grab_cursor();
                    self.cursor_grabbed = true;
                }
            }
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    Runner::run(MyGame::new())
}
