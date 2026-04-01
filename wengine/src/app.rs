use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::Window,
};

use crate::{
    EngineState,
    scene::{EngineEvent, Scene, SceneContext, SceneInstance},
};

struct App {
    state: Option<EngineState>,
    scenes: Vec<SceneInstance>,
    last_frame_time: std::time::Instant,
    physics_update: f32,
    clock: f32,
    width: u32,
    height: u32,
    title: String,
    fullscreen: bool,
    resizable: bool,
}

// Based on the window "event" or action we run the correct function in game.

impl ApplicationHandler<EngineState> for App {
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state: &mut EngineState = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::RedrawRequested => {
                let current_time = std::time::Instant::now();
                let delta = (current_time - self.last_frame_time).as_secs_f32();
                self.last_frame_time = current_time;
                self.clock += delta;

                // We run the physics at a set time but we render at the monitors FPS.
                if self.clock >= self.physics_update {
                    let (new_collisions, removed_collision) = state.update();

                    {
                        for i in 0..self.scenes.len() {
                            let mut scene = self.scenes.remove(i);

                            if !scene.is_active {
                                let mut scene_context = SceneContext::new(state, &mut self.scenes);
                                scene.scene.on_init(&mut scene_context);

                                scene.is_active = true;
                            }

                            let mut scene_context = SceneContext::new(state, &mut self.scenes);
                            scene.scene.on_update(delta, &mut scene_context);

                            self.scenes.insert(i, scene);
                        }
                    }

                    for pair in new_collisions {
                        for i in 0..self.scenes.len() {
                            // We send the event twice to rapresent both entities perspective

                            let mut scene = self.scenes.remove(i);

                            let mut scene_context = SceneContext::new(state, &mut self.scenes);
                            scene.scene.on_event(
                                EngineEvent::CollisionEnter {
                                    entity: pair.0.clone(),
                                    other: pair.1.clone(),
                                },
                                &mut scene_context,
                            );

                            let mut scene_context = SceneContext::new(state, &mut self.scenes);

                            scene.scene.on_event(
                                EngineEvent::CollisionEnter {
                                    entity: pair.1.clone(),
                                    other: pair.0.clone(),
                                },
                                &mut scene_context,
                            );

                            self.scenes.insert(i, scene);
                        }
                    }

                    for pair in removed_collision {
                        for i in 0..self.scenes.len() {
                            let mut scene = self.scenes.remove(i);

                            let mut scene_context = SceneContext::new(state, &mut self.scenes);
                            scene.scene.on_event(
                                EngineEvent::CollisionExit {
                                    entity: pair.0.clone(),
                                    other: pair.1.clone(),
                                },
                                &mut scene_context,
                            );

                            let mut scene_context = SceneContext::new(state, &mut self.scenes);
                            scene.scene.on_event(
                                EngineEvent::CollisionExit {
                                    entity: pair.1.clone(),
                                    other: pair.0.clone(),
                                },
                                &mut scene_context,
                            );

                            self.scenes.insert(i, scene);
                        }
                    }

                    self.clock -= self.physics_update;
                }

                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.renderer.resize(size.width, size.height);
                    }
                    Err(e) => {
                        log::error!("Unable to render {}", e);
                    }
                }
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.renderer.resize(size.width, size.height),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key,
                        state: key_state,
                        ..
                    },
                ..
            } => {
                for i in 0..self.scenes.len() {
                    let mut scene = self.scenes.remove(i);
                    let mut scene_context = SceneContext::new(state, &mut self.scenes);
                    scene.scene.on_event(
                        EngineEvent::Key {
                            physical_key,
                            pressed: key_state.is_pressed(),
                        },
                        &mut scene_context,
                    );
                    self.scenes.insert(i, scene);
                }
            }

            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => {
                for i in 0..self.scenes.len() {
                    let mut scene = self.scenes.remove(i);
                    let mut scene_context = SceneContext::new(state, &mut self.scenes);
                    scene.scene.on_event(
                        EngineEvent::MouseButton {
                            button,
                            pressed: button_state == ElementState::Pressed,
                        },
                        &mut scene_context,
                    );
                    self.scenes.insert(i, scene);
                }
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let state: &mut EngineState = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        if let winit::event::DeviceEvent::MouseMotion { delta } = event {
            if state.cursor_grabbed {
                for i in 0..self.scenes.len() {
                    let mut scene = self.scenes.remove(i);
                    let mut scene_context = SceneContext::new(state, &mut self.scenes);
                    scene.scene.on_event(
                        EngineEvent::MouseMotion {
                            delta_x: delta.0,
                            delta_y: delta.1,
                        },
                        &mut scene_context,
                    );
                    self.scenes.insert(i, scene);
                }
            }
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        use winit::window::Fullscreen;

        let mut window_attributes = Window::default_attributes()
            .with_title(&self.title)
            .with_inner_size(winit::dpi::PhysicalSize::new(self.width, self.height))
            .with_resizable(self.resizable);

        if self.fullscreen {
            window_attributes =
                window_attributes.with_fullscreen(Some(Fullscreen::Borderless(None)));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        let state = pollster::block_on(EngineState::new(window.clone())).unwrap();

        self.state = Some(state);
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: EngineState) {
        self.state = Some(event);
    }
}

// Public API

pub struct Runner {
    main: SceneInstance,
    width: u32,
    height: u32,
    title: String,
    fullscreen: bool,
    resizable: bool,
}

impl Runner {
    pub fn new(main: impl Scene + 'static) -> Self {
        Self {
            main: SceneInstance {
                scene: Box::new(main),
                is_active: false,
            },
            width: 800,
            height: 600,
            title: "WEngine Game".to_string(),
            fullscreen: false,
            resizable: true,
        }
    }

    pub fn window_width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }

    pub fn window_height(mut self, height: u32) -> Self {
        self.height = height;
        self
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    pub fn fullscreen(mut self, fullscreen: bool) -> Self {
        self.fullscreen = fullscreen;
        self
    }

    pub fn resizable(mut self, resizable: bool) -> Self {
        self.resizable = resizable;
        self
    }

    pub fn run(self) -> anyhow::Result<()> {
        env_logger::init();

        let event_loop = EventLoop::with_user_event().build()?;
        let mut app = App {
            state: None,
            scenes: vec![self.main],
            last_frame_time: std::time::Instant::now(),
            clock: 0.0,
            physics_update: 1.0 / 60.0,
            width: self.width,
            height: self.height,
            title: self.title,
            fullscreen: self.fullscreen,
            resizable: self.resizable,
        };

        event_loop.run_app(&mut app)?;

        Ok(())
    }
}
