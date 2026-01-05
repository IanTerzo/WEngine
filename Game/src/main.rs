use nalgebra::{self, vector, UnitQuaternion, Vector3};
use rand::random_range;
use winit::keyboard::{KeyCode, PhysicalKey};
use WEngine::{
    entity::{CameraState, Entity, EntityHandle, EntityState},
    model::MeshHandle,
    EngineEvent, Game, Runner, Scene, Transform,
};

struct CameraController {
    sensitivity: f32,
    yaw: f32,
    pitch: f32,
}

impl CameraController {
    fn new() -> Self {
        Self {
            sensitivity: 0.002,
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    fn process_mouse(&mut self, delta_x: f64, delta_y: f64) {
        self.yaw -= delta_x as f32 * self.sensitivity;
        self.pitch -= delta_y as f32 * self.sensitivity;

        // Clamp pitch to prevent camera flipping, but allow yaw to wrap around
        self.pitch = self
            .pitch
            .clamp(-89.0f32.to_radians(), 89.0f32.to_radians());

        // Normalize yaw to keep it within 0 to 2π (optional, prevents float overflow)
        use std::f32::consts::TAU;
        self.yaw = self.yaw.rem_euclid(TAU);
    }

    fn get_rotation(&self) -> UnitQuaternion<f32> {
        let yaw_quat = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), self.yaw);
        let pitch_quat = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), self.pitch);
        yaw_quat * pitch_quat
    }
}

struct PlayerController {
    speed: f32,
    jump_force: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl PlayerController {
    fn new(speed: f32) -> Self {
        Self {
            speed,
            jump_force: 40.0,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    fn get_movement_direction(&self, rotation: UnitQuaternion<f32>) -> Vector3<f32> {
        let mut velocity = vector![0.0, 0.0, 0.0];

        if self.is_forward_pressed {
            velocity += get_forward(rotation);
        }
        if self.is_backward_pressed {
            velocity -= get_forward(rotation);
        }
        if self.is_left_pressed {
            velocity += get_left(rotation);
        }
        if self.is_right_pressed {
            velocity += get_right(rotation);
        }

        // Normalize to prevent faster diagonal movement
        if velocity.magnitude() > 0.0 {
            velocity = velocity.normalize() * self.speed;
        }

        velocity
    }
}

struct MyGame {
    camera_controller: CameraController,
    player_controller: PlayerController,
    cube_mesh: Option<MeshHandle>,
    player_entity: Option<EntityHandle>,
    cursor_grabbed: bool,
}

impl MyGame {
    fn new() -> Self {
        Self {
            camera_controller: CameraController::new(),
            player_controller: PlayerController::new(10.0), // Default speed of 5.0
            cube_mesh: None,
            player_entity: None,
            cursor_grabbed: false,
        }
    }
}

fn get_forward(rotation: UnitQuaternion<f32>) -> Vector3<f32> {
    rotation * vector![0.0, 0.0, -1.0]
}

fn get_right(rotation: UnitQuaternion<f32>) -> Vector3<f32> {
    rotation * vector![1.0, 0.0, 0.0]
}

fn get_left(rotation: UnitQuaternion<f32>) -> Vector3<f32> {
    rotation * vector![-1.0, 0.0, 0.0]
}

impl Game for MyGame {
    fn on_init(&mut self, scene: &mut Scene) {
        let cube_mesh = scene
            .load_obj("/home/ianterzo/Work/WEngine/res/cube.obj")
            .unwrap()[0];

        // Ground
        scene.spawn(
            Entity::static_body(Transform {
                position: vector![0.0, -30.0, 0.0],
                rotation: UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.0f32.to_radians())
                    .into_inner(),
                scale: vector![10.0, 1.0, 10.0],
            })
            .collider_cuboid(vector![10.0, 1.0, 10.0])
            .mesh(cube_mesh),
        );

        // Player with camera
        let player = scene.spawn(
            Entity::dynamic_body(Transform {
                position: vector![0.0, 0.0, 0.0],
                rotation: UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.0f32.to_radians())
                    .into_inner(),
                scale: vector![1.0, 1.0, 1.0],
            })
            .add_child(Entity::camera(Transform {
                position: vector![0.0, 2.0, 0.0],
                rotation: UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.0f32.to_radians())
                    .into_inner(),
                scale: vector![1.0, 1.0, 1.0],
            }))
            .add_child(Entity::mesh_instance(
                cube_mesh,
                Transform {
                    position: vector![2.0, 1.0, 0.0],
                    rotation: UnitQuaternion::from_axis_angle(
                        &Vector3::y_axis(),
                        30.0f32.to_radians(),
                    )
                    .into_inner(),
                    scale: vector![0.5, 0.5, 0.5],
                },
            ))
            .collider_cuboid(vector![1.0, 1.0, 1.0]),
        );

        self.player_entity = Some(player);
        self.cube_mesh = Some(cube_mesh);

        // Grab cursor for mouse look
        scene.grab_cursor();
        self.cursor_grabbed = true;
    }

    fn on_update(&mut self, _delta: f32, scene: &mut Scene) {
        // Update camera rotation based on mouse input
        if let Some(player_handle) = &self.player_entity {
            if let EntityState::Camera(camera_state) =
                &mut scene.get_entity(player_handle.clone()).children[0].state
            {
                *camera_state = CameraState {
                    transform: Transform {
                        position: camera_state.transform.position,
                        rotation: self.camera_controller.get_rotation().into_inner(),
                        scale: camera_state.transform.scale,
                    },
                    fov: camera_state.fov,
                    near: camera_state.near,
                    far: camera_state.far,
                };
            }

            // Calculate movement velocity
            let rot = self.camera_controller.get_rotation();
            let velocity = self.player_controller.get_movement_direction(rot);

            // Get current velocity to preserve Y component (gravity/jumping)
            let current_vel = scene.get_linvel(player_handle.clone());

            // Set velocity - preserve Y for gravity, replace X and Z for movement
            scene.set_linvel(
                player_handle.clone(),
                vector![velocity.x, current_vel.y, velocity.z],
            );
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
                        self.player_controller.is_forward_pressed = pressed;
                    }
                    KeyCode::KeyA | KeyCode::ArrowLeft => {
                        self.player_controller.is_left_pressed = pressed;
                    }
                    KeyCode::KeyS | KeyCode::ArrowDown => {
                        self.player_controller.is_backward_pressed = pressed;
                    }
                    KeyCode::KeyD | KeyCode::ArrowRight => {
                        self.player_controller.is_right_pressed = pressed;
                    }
                    KeyCode::Space => {
                        if pressed {
                            if let Some(player_handle) = &self.player_entity {
                                // Apply upward impulse for jumping
                                scene.apply_impulse(
                                    player_handle.clone(),
                                    vector![0.0, self.player_controller.jump_force, 0.0],
                                );
                            }
                        }
                    }
                    KeyCode::KeyQ => {
                        if !pressed {
                            return;
                        }

                        if let Some(cube_mesh) = self.cube_mesh {
                            scene.spawn(
                                Entity::dynamic_body(Transform {
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
                                })
                                .mesh(cube_mesh)
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
            EngineEvent::MouseButton { button: _, pressed } => {
                // Optional: grab cursor on mouse click
                if pressed && !self.cursor_grabbed {
                    scene.grab_cursor();
                    self.cursor_grabbed = true;
                }
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    Runner::new(MyGame::new())
        .fullscreen(true)
        .window_width(1280)
        .window_height(720)
        .title("Fixed Size Game")
        .run()
}
