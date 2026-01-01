use Engine::{State, Transform, Vertex};
use cgmath::Rotation3;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::PhysicalKey,
    window::Window,
};

pub struct App {
    state: Option<State>,
}

impl App {
    pub fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler<State> for App {
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.resize(size.width, size.height);
                    }
                    Err(e) => {
                        log::error!("Unable to render {}", e);
                    }
                }
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                state.render();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state.is_pressed()),
            _ => {}
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        let mut state = pollster::block_on(State::new(window.clone())).unwrap();

        let vertices: &[Vertex] = &[
            // front
            Vertex {
                position: [-1.0, -1.0, 1.0],
                tex_coords: [0.0, 0.0],
            },
            Vertex {
                position: [1.0, -1.0, 1.0],
                tex_coords: [1.0, 0.0],
            },
            Vertex {
                position: [1.0, 1.0, 1.0],
                tex_coords: [1.0, 1.0],
            },
            Vertex {
                position: [-1.0, 1.0, 1.0],
                tex_coords: [0.0, 1.0],
            },
            // back
            Vertex {
                position: [-1.0, -1.0, -1.0],
                tex_coords: [1.0, 0.0],
            },
            Vertex {
                position: [1.0, -1.0, -1.0],
                tex_coords: [0.0, 0.0],
            },
            Vertex {
                position: [1.0, 1.0, -1.0],
                tex_coords: [0.0, 1.0],
            },
            Vertex {
                position: [-1.0, 1.0, -1.0],
                tex_coords: [1.0, 1.0],
            },
        ];

        let indices: &[u16] = &[
            0, 1, 2, 2, 3, 0, // front
            1, 5, 6, 6, 2, 1, // right
            5, 4, 7, 7, 6, 5, // back
            4, 0, 3, 3, 7, 4, // left
            3, 2, 6, 6, 7, 3, // top
            4, 5, 1, 1, 0, 4, // bottom
        ];

        let material = state.load_material();

        let mesh = state.add_mesh(vertices, indices, material);

        state.add_instance(
            mesh,
            Transform {
                position: cgmath::vec3(0.0, 0.0, -2.0), // 2 units in front of camera
                rotation: cgmath::Quaternion::from_angle_y(cgmath::Deg(0.0)), // no rotation
                scale: cgmath::vec3(1.0, 1.0, 1.0),     // normal size
            },
        );

        state.add_instance(
            mesh,
            Transform {
                position: cgmath::vec3(0.0, 2.0, -2.0), // 2 units in front of camera
                rotation: cgmath::Quaternion::from_angle_y(cgmath::Deg(0.0)), // no rotation
                scale: cgmath::vec3(0.5, 0.5, 0.5),     // normal size
            },
        );

        // Store state
        self.state = Some(state);
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        self.state = Some(event);
    }
}

pub fn main() -> anyhow::Result<()> {
    env_logger::init();

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}
