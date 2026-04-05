use wengine::{
    entity::{EntityHandle, builder::EntityBuilder, refs::EntityRef},
    scene::{EngineEvent, Scene, SceneContext},
    transform::Transform,
};

use nalgebra::{self, UnitQuaternion, Vector3, vector};
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

pub struct Player {
    camera_controller: CameraController,
    player_controller: PlayerController,
    player_handle: Option<EntityHandle>,
    cursor_grabbed: bool,
}

impl Player {
    pub fn new() -> Self {
        Self {
            camera_controller: CameraController::new(),
            player_controller: PlayerController::new(),
            player_handle: None,
            cursor_grabbed: false,
        }
    }
}

impl Scene for Player {
    fn on_init(&mut self, ctx: &mut SceneContext) {
        let player_handle = ctx
            .spawn(
                EntityBuilder::dynamic_body(Transform {
                    position: vector![0.0, 0.0, 0.0],
                    rotation: UnitQuaternion::identity(),
                    scale: vector![1.0, 1.0, 1.0],
                })
                .add_child(
                    EntityBuilder::empty(Transform {
                        position: vector![0.0, 1.8, 0.0],
                        rotation: UnitQuaternion::identity(),
                        scale: vector![1.0, 1.0, 1.0],
                    })
                    .add_child(
                        EntityBuilder::camera(Transform {
                            position: vector![0.0, 0.0, 0.0],
                            rotation: UnitQuaternion::identity(),
                            scale: vector![1.0, 1.0, 1.0],
                        })
                        .fov(80.0),
                    ),
                )
                .collider_capsule(0.9, 0.5)
                .tag("player_body")
                .gravity_scale(3.5),
            )
            .unwrap();

        ctx.get_entity(player_handle.clone())
            .unwrap()
            .into_dynamicbody()
            .unwrap()
            .set_enabled_rotations(false, false, false)
            .unwrap();

        self.player_handle = Some(player_handle);

        ctx.grab_cursor();
        self.cursor_grabbed = true;
    }

    fn on_physics_update(&mut self, delta: f32, ctx: &mut SceneContext) {
        let Some(player_handle) = self.player_handle.clone() else {
            return;
        };

        let mut player = ctx
            .get_entity(player_handle.clone())
            .unwrap()
            .into_dynamicbody()
            .unwrap();

        // Jump logic

        if self.player_controller.is_jump_pressed && self.player_controller.is_on_ground {
            let current_vel = player.get_linvel().unwrap();
            player
                .set_linvel(vector![
                    current_vel.x,
                    self.player_controller.jump_velocity,
                    current_vel.z
                ])
                .unwrap();
        }

        // Movement logic

        let rot = self.camera_controller.get_rotation();
        let desired_velocity = self.player_controller.get_movement_direction(rot);
        let current_vel = player.get_linvel().unwrap();

        if self.player_controller.is_on_ground {
            player
                .set_linvel(vector![
                    desired_velocity.x,
                    current_vel.y,
                    desired_velocity.z
                ])
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

            player
                .set_linvel(vector![new_horizontal.x, current_vel.y, new_horizontal.z])
                .unwrap();
        }

        // Respawn logic

        let pos = player.get_position().unwrap();

        if pos.y < -100.0 {
            player.set_position(vector![0.0, 0.0, 0.0]).unwrap();
            player.set_angvel(vector![0.0, 0.0, 0.0]).unwrap();
            player.set_linvel(vector![0.0, 0.0, 0.0]).unwrap();
        }

        // Update view rotation

        if let EntityRef::Empty(view) = &mut player.get_child(0).unwrap() {
            view.entity.transform.rotation = self.camera_controller.get_rotation()
        }
    }

    fn on_event(&mut self, event: EngineEvent, ctx: &mut SceneContext) {
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
                    KeyCode::Escape => {
                        ctx.release_cursor();
                        self.cursor_grabbed = false;
                    }
                    _ => {}
                },
                _ => {}
            },
            EngineEvent::MouseButton { button: _, pressed } => {
                if pressed && !self.cursor_grabbed {
                    ctx.grab_cursor();
                    self.cursor_grabbed = true;
                }
            }
            EngineEvent::MouseMotion { delta_x, delta_y } => {
                if self.cursor_grabbed {
                    self.camera_controller.process_mouse(delta_x, delta_y);
                }
            }
            EngineEvent::CollisionEnter { entity, other } => {
                let tag_self = match ctx.get_entity(entity).unwrap() {
                    EntityRef::DynamicBody(dynamic_body) => dynamic_body.entity.tag.clone(),
                    _ => Some("".to_string()),
                };

                let tag_other = match ctx.get_entity(other).unwrap() {
                    EntityRef::StaticBody(static_body) => static_body.entity.tag.clone(),
                    EntityRef::DynamicBody(dynamic_body) => dynamic_body.entity.tag.clone(),
                    _ => Some("".to_string()),
                };

                if tag_self.as_deref() == Some("player_body")
                    && tag_other.as_deref() == Some("walkable")
                {
                    self.player_controller.is_on_ground = true;
                }
            }

            EngineEvent::CollisionExit { entity, other } => {
                let tag_self = match ctx.get_entity(entity).unwrap() {
                    EntityRef::DynamicBody(dynamic_body) => dynamic_body.entity.tag.clone(),
                    _ => Some("".to_string()),
                };

                let tag_other = match ctx.get_entity(other).unwrap() {
                    EntityRef::StaticBody(static_body) => static_body.entity.tag.clone(),
                    EntityRef::DynamicBody(dynamic_body) => dynamic_body.entity.tag.clone(),
                    _ => Some("".to_string()),
                };

                if tag_self.as_deref() == Some("player_body")
                    && tag_other.as_deref() == Some("walkable")
                {
                    self.player_controller.is_on_ground = false;
                }
            }
        }
    }
}
