use WEngine::{
    EngineEvent, Game, Runner, Scene, Transform,
    entity::{Entity, EntityBuilder, EntityHandle},
    model::MeshHandle,
};
use nalgebra::{self, UnitQuaternion, Vector3, vector};
use rand::random_range;
use winit::keyboard::{KeyCode, PhysicalKey};

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
    jump_velocity: f32,
    air_control_factor: f32,
    is_on_ground: bool,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_jump_pressed: bool,
}

fn get_forward(rotation: UnitQuaternion<f32>) -> Vector3<f32> {
    rotation * Vector3::new(0.0, 0.0, -1.0)
}

fn get_right(rotation: UnitQuaternion<f32>) -> Vector3<f32> {
    rotation * vector![1.0, 0.0, 0.0]
}

impl PlayerController {
    fn new() -> Self {
        Self {
            speed: 15.0,
            jump_velocity: 12.0,
            air_control_factor: 0.5,
            is_on_ground: false,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_jump_pressed: false,
        }
    }

    fn get_movement_direction(&self, rotation: UnitQuaternion<f32>) -> Vector3<f32> {
        let mut velocity = Vector3::new(0.0, 0.0, 0.0);

        let mut forward = get_forward(rotation);
        let mut right = get_right(rotation);

        forward.y = 0.0;
        right.y = 0.0;

        if forward.magnitude_squared() > 0.0 {
            forward = forward.normalize();
        }
        if right.magnitude_squared() > 0.0 {
            right = right.normalize();
        }

        if self.is_forward_pressed {
            velocity += forward;
        }
        if self.is_backward_pressed {
            velocity -= forward;
        }
        if self.is_left_pressed {
            velocity -= right;
        }
        if self.is_right_pressed {
            velocity += right;
        }

        // Normalize to prevent faster diagonal movement
        if velocity.magnitude_squared() > 0.0 {
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
            player_controller: PlayerController::new(),
            cube_mesh: None,
            player_entity: None,
            cursor_grabbed: false,
        }
    }
}

impl Game for MyGame {
    fn on_init(&mut self, scene: &mut Scene) {
        let cube_mesh = scene.load_obj("../res/cube.obj").unwrap()[0];

        // Ground

        scene.spawn(
            EntityBuilder::static_body(Transform {
                position: vector![0.0, -30.0, 0.0],
                rotation: UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.0f32.to_radians())
                    .into_inner(),
                scale: vector![10.0, 1.0, 10.0],
            })
            .collider_cuboid(vector![10.0, 1.0, 10.0])
            .mesh(cube_mesh)
            .tag("walkable"),
        );

        // Player

        let player_handle = scene.spawn(
            EntityBuilder::dynamic_body(Transform {
                position: vector![0.0, 0.0, 0.0],
                rotation: UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.0f32.to_radians())
                    .into_inner(),
                scale: vector![1.0, 1.0, 1.0],
            })
            .add_child(
                EntityBuilder::camera(Transform {
                    position: vector![0.0, 1.8, 0.0],
                    rotation: UnitQuaternion::from_axis_angle(
                        &Vector3::y_axis(),
                        0.0f32.to_radians(),
                    )
                    .into_inner(),
                    scale: vector![1.0, 1.0, 1.0],
                })
                .fov(80.0),
            )
            .collider_capsule(0.9, 0.5)
            .tag("player_body")
            .gravity_scale(3.5), // Feeles more natural.
        );

        scene.set_enabled_rotations(player_handle.clone(), false, false, false);

        self.player_entity = Some(player_handle);
        self.cube_mesh = Some(cube_mesh);

        scene.grab_cursor();
        self.cursor_grabbed = true;
    }

    fn on_update(&mut self, delta: f32, scene: &mut Scene) {
        if let Some(player_handle) = &self.player_entity {
            // Handle jumping in update loop instead of event
            if self.player_controller.is_jump_pressed && self.player_controller.is_on_ground {
                let current_vel = scene.get_linvel(player_handle.clone()).unwrap();
                scene
                    .set_linvel(
                        player_handle.clone(),
                        vector![
                            current_vel.x,
                            self.player_controller.jump_velocity,
                            current_vel.z
                        ],
                    )
                    .unwrap();
            }

            // Update camera rotation
            if let Entity::DynamicBody(rigid_body) =
                &mut scene.get_entity(player_handle.clone()).unwrap()
            {
                if let Entity::Camera(view) = &mut rigid_body.children[0] {
                    view.transform.rotation = self.camera_controller.get_rotation().into_inner()
                }
            }

            // Movement logic
            let rot = self.camera_controller.get_rotation();
            let desired_velocity = self.player_controller.get_movement_direction(rot);
            let current_vel = scene.get_linvel(player_handle.clone()).unwrap();

            if self.player_controller.is_on_ground {
                scene
                    .set_linvel(
                        player_handle.clone(),
                        vector![desired_velocity.x, current_vel.y, desired_velocity.z],
                    )
                    .unwrap();
            } else {
                let current_horizontal = vector![current_vel.x, 0.0, current_vel.z];
                let adjustment = vector![
                    desired_velocity.x - current_vel.x,
                    0.0,
                    desired_velocity.z - current_vel.z
                ];
                let air_influence = self.player_controller.air_control_factor * delta * 20.0;
                let new_horizontal = current_horizontal + adjustment * air_influence;

                scene
                    .set_linvel(
                        player_handle.clone(),
                        vector![new_horizontal.x, current_vel.y, new_horizontal.z],
                    )
                    .unwrap();
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
                        self.player_controller.is_jump_pressed = pressed;
                    }
                    KeyCode::KeyQ => {
                        if !pressed {
                            return;
                        }

                        if let Some(cube_mesh) = self.cube_mesh {
                            scene.spawn(
                                EntityBuilder::dynamic_body(Transform {
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
                                .collider_cuboid(vector![1.0, 1.0, 1.0])
                                .tag("walkable"),
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
            EngineEvent::CollisionEnter { entity, other } => {
                let tag_name: Option<String> =
                    if let Entity::DynamicBody(static_body) = scene.get_entity(entity).unwrap() {
                        static_body.tag.clone()
                    } else {
                        None
                    };

                let tag_name_other: Option<String> = if let Entity::StaticBody(static_body) =
                    scene.get_entity(other.clone()).unwrap()
                {
                    static_body.tag.clone()
                } else if let Entity::DynamicBody(dynamic_body) = scene.get_entity(other).unwrap() {
                    dynamic_body.tag.clone()
                } else {
                    None
                };

                if tag_name.unwrap_or_default() == "player_body"
                    && tag_name_other.unwrap_or_default() == "walkable"
                {
                    self.player_controller.is_on_ground = true;
                }
            }
            EngineEvent::CollisionExit { entity, other } => {
                let tag_name: Option<String> =
                    if let Entity::DynamicBody(static_body) = scene.get_entity(entity).unwrap() {
                        static_body.tag.clone()
                    } else {
                        None
                    };

                let tag_name_other: Option<String> = if let Entity::StaticBody(static_body) =
                    scene.get_entity(other.clone()).unwrap()
                {
                    static_body.tag.clone()
                } else if let Entity::DynamicBody(dynamic_body) = scene.get_entity(other).unwrap() {
                    dynamic_body.tag.clone()
                } else {
                    None
                };

                if tag_name.unwrap_or_default() == "player_body"
                    && tag_name_other.unwrap_or_default() == "walkable"
                {
                    self.player_controller.is_on_ground = false;
                }
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    Runner::new(MyGame::new())
        .window_width(1280)
        .window_height(720)
        .title("Fixed Size Game")
        .run()
}
