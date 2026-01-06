use crate::{
    entity::{
        Entity, EntityHandle, EntityRef, OPENGL_TO_WGPU_MATRIX, get_entity_from_handle, spawn,
    },
    model::{InstanceRaw, MeshData, MeshHandle, ModelVertex, load_obj},
    physics::{
        PhysicsWorld, add_force, apply_impulse, get_angvel, get_linvel, set_angvel,
        set_enabled_rotations, set_linvel,
    },
};
use nalgebra::{
    self, Isometry, Matrix4, Perspective3, Quaternion, Translation3, UnitQuaternion, Vector3,
};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::PhysicalKey,
    window::Window,
};

pub mod entity;
pub mod model;
pub mod physics;
pub mod texture;

const MAX_INSTANCES: usize = 100;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transform {
    pub position: Vector3<f32>,
    pub rotation: Quaternion<f32>,
    pub scale: Vector3<f32>,
}

impl Transform {
    pub fn to_matrix(&self) -> Matrix4<f32> {
        let translation = Translation3::from(self.position).to_homogeneous();
        // make sure the quaternion is treated as a rotation
        let rotation = UnitQuaternion::from_quaternion(self.rotation).to_homogeneous();
        let scale = Matrix4::new_nonuniform_scaling(&self.scale);

        translation * rotation * scale
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct InstanceHandle {
    pub mesh: MeshHandle,
    pub instance_index: usize,
}

pub struct Instance {
    pub transform: Transform,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

pub struct EngineState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    render_pipeline: wgpu::RenderPipeline,
    pub window: Arc<Window>,
    depth_texture: texture::Texture,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    meshes: Vec<MeshData>,
    camera_uniform: CameraUniform,
    physics_world: PhysicsWorld,
    entities: Vec<EntityRef>,
    cursor_grabbed: bool,
}

impl EngineState {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<EngineState> {
        // Creates the wgpu instance and the SurfaceConfiguration

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap();

        let size = window.inner_size();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        // Get device and command queue

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await?;

        // Creates necessary buffer and bind group for the camera

        let camera_uniform = CameraUniform {
            view_proj: Matrix4::identity().into(),
        };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&camera_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: None,
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: None,
        });

        // Create the texture bind group layout

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        // Create shader module

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // Create depth texture

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth-texture");

        // Create render pipeline

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[ModelVertex::desc(), InstanceRaw::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let physics_world = PhysicsWorld::new();

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            render_pipeline,
            window,
            depth_texture,
            camera_buffer,
            camera_bind_group,
            texture_bind_group_layout,
            meshes: vec![],
            camera_uniform,
            physics_world,
            entities: vec![],
            cursor_grabbed: false,
        })
    }

    pub fn load_obj(&mut self, path: &str) -> anyhow::Result<Vec<MeshHandle>> {
        load_obj(
            &self.device,
            &self.queue,
            &self.texture_bind_group_layout,
            path,
            &mut self.meshes,
        )
    }

    pub fn get_entity(&mut self, entity_handle: EntityHandle) -> &mut EntityRef {
        get_entity_from_handle(&mut self.entities, entity_handle)
    }

    pub fn spawn(&mut self, entity: impl Into<Entity>) -> EntityHandle {
        spawn(
            &mut self.entities,
            &mut self.meshes,
            &mut self.physics_world,
            &self.queue,
            &mut self.camera_uniform,
            &self.camera_buffer,
            &self.config,
            entity,
        )
    }

    // Rigidbody functions

    pub fn apply_impulse(&mut self, entity_handle: EntityHandle, vector: Vector3<f32>) {
        apply_impulse(
            &mut self.physics_world,
            &mut self.entities,
            entity_handle,
            vector,
        );
    }

    pub fn add_force(&mut self, entity_handle: EntityHandle, vector: Vector3<f32>) {
        add_force(
            &mut self.physics_world,
            &mut self.entities,
            entity_handle,
            vector,
        );
    }

    pub fn get_linvel(&mut self, entity_handle: EntityHandle) -> Vector3<f32> {
        get_linvel(&mut self.physics_world, &mut self.entities, entity_handle)
    }

    pub fn set_linvel(&mut self, entity_handle: EntityHandle, vector: Vector3<f32>) {
        set_linvel(
            &mut self.physics_world,
            &mut self.entities,
            entity_handle,
            vector,
        );
    }

    pub fn get_angvel(&mut self, entity_handle: EntityHandle) -> Vector3<f32> {
        get_angvel(&mut self.physics_world, &mut self.entities, entity_handle)
    }

    pub fn set_angvel(&mut self, entity_handle: EntityHandle, vector: Vector3<f32>) {
        set_angvel(
            &mut self.physics_world,
            &mut self.entities,
            entity_handle,
            vector,
        );
    }

    pub fn set_enabled_rotations(
        &mut self,
        entity_handle: EntityHandle,
        enable_x: bool,
        enable_y: bool,
        enable_z: bool,
    ) {
        set_enabled_rotations(
            &mut self.physics_world,
            &mut self.entities,
            entity_handle,
            enable_x,
            enable_y,
            enable_z,
        );
    }

    // Instances    TODO: Instance rework

    pub fn get_instance(&self, handle: InstanceHandle) -> Option<&Instance> {
        self.meshes
            .get(handle.mesh.0)?
            .instances
            .get(handle.instance_index)
    }

    pub fn update_instance(&mut self, handle: InstanceHandle, transform: Transform) {
        if let Some(mesh_data) = self.meshes.get_mut(handle.mesh.0) {
            if let Some(instance) = mesh_data.instances.get_mut(handle.instance_index) {
                instance.transform = transform;

                let instance_raw = instance.to_raw();
                let offset = handle.instance_index * std::mem::size_of::<InstanceRaw>();

                self.queue.write_buffer(
                    &mesh_data.instance_buffer,
                    offset as wgpu::BufferAddress,
                    bytemuck::cast_slice(&[instance_raw]),
                );
            }
        }
    }

    // Events

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth");
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
        }
    }

    pub fn grab_cursor(&mut self) {
        self.cursor_grabbed = true;
        let _ = self
            .window
            .set_cursor_grab(winit::window::CursorGrabMode::Confined);
        self.window.set_cursor_visible(false);
    }

    pub fn release_cursor(&mut self) {
        self.cursor_grabbed = false;
        let _ = self
            .window
            .set_cursor_grab(winit::window::CursorGrabMode::None);
        self.window.set_cursor_visible(true);
    }

    pub fn rigid_body_trickle_down_update(
        &mut self,
        entity_ref: &EntityRef,
        parent_position: Vector3<f32>,
        parent_rotation: Quaternion<f32>,
    ) {
        // We want to update all non rigidbody children with the physics of the parent rigidbody.
        match &entity_ref {
            EntityRef::DynamicBody(entity)
            | EntityRef::StaticBody(entity)
            | EntityRef::KinematicBody(entity) => {
                let rigid_body_calc = self
                    .physics_world
                    .rigid_body_set
                    .get(entity.rigid_body_handle)
                    .unwrap();

                let iso = rigid_body_calc.position();
                let position: Vector3<f32> = iso.translation.vector;
                let rotation = iso.rotation.into_inner();

                // We apply the physics of the parent rigidbody on all children
                for child in &entity.children {
                    self.rigid_body_trickle_down_update(child, position, rotation);
                }

                if let Some(instance_handle) = entity.instance_handle {
                    let scale = self.get_instance(instance_handle).unwrap().transform.scale;

                    self.update_instance(
                        instance_handle,
                        Transform {
                            position,
                            rotation,
                            scale,
                        },
                    );
                }
            }
            EntityRef::MeshInstance(entity) => {
                let rotated_offset = UnitQuaternion::from_quaternion(parent_rotation)
                    .transform_vector(&entity.transform.position);

                let new_position = parent_position + rotated_offset;

                let new_rotation = (UnitQuaternion::from_quaternion(parent_rotation)
                    * UnitQuaternion::from_quaternion(entity.transform.rotation))
                .into_inner();

                for child in &entity.children {
                    self.rigid_body_trickle_down_update(child, new_position, new_rotation);
                }

                self.update_instance(
                    entity.instance_handle,
                    Transform {
                        position: new_position,
                        rotation: new_rotation,
                        scale: entity.transform.scale,
                    },
                );
            }
            EntityRef::Camera(entity) => {
                let iso = Isometry::from_parts(
                    Translation3::from(parent_position + entity.transform.position),
                    UnitQuaternion::from_quaternion(parent_rotation)
                        * UnitQuaternion::from_quaternion(entity.transform.rotation),
                );

                let view = iso.inverse().to_homogeneous();

                let aspect = self.config.width as f32 / self.config.height as f32;

                let proj =
                    Perspective3::new(aspect, entity.fov.to_radians(), entity.near, entity.far)
                        .to_homogeneous();

                self.camera_uniform.view_proj = (OPENGL_TO_WGPU_MATRIX * proj * view).into();

                self.queue.write_buffer(
                    &self.camera_buffer,
                    0,
                    bytemuck::cast_slice(&[self.camera_uniform]),
                );
            }
            EntityRef::Empty(entity) => {
                let rotated_offset = UnitQuaternion::from_quaternion(parent_rotation)
                    .transform_vector(&entity.transform.position);

                let new_position = parent_position + rotated_offset;

                let new_rotation = (UnitQuaternion::from_quaternion(parent_rotation)
                    * UnitQuaternion::from_quaternion(entity.transform.rotation))
                .into_inner();

                for child in &entity.children {
                    self.rigid_body_trickle_down_update(child, new_position, new_rotation);
                }
            }
        }
    }

    pub fn update(&mut self) {
        self.physics_world.step();

        for i in 0..self.entities.len() {
            let (position, rotation, instance_handle, children) = {
                let entity_info = &mut self.entities[i];

                match entity_info {
                    EntityRef::DynamicBody(rigid_body)
                    | EntityRef::StaticBody(rigid_body)
                    | EntityRef::KinematicBody(rigid_body) => {
                        let handle = rigid_body.rigid_body_handle; // assumed Copy
                        let rigid_body_calc =
                            self.physics_world.rigid_body_set.get(handle).unwrap();

                        let iso = rigid_body_calc.position();
                        let position: Vector3<f32> = iso.translation.vector;
                        let rotation = iso.rotation.into_inner();

                        let children = std::mem::take(&mut rigid_body.children);

                        (position, rotation, rigid_body.instance_handle, children)
                    }
                    _ => continue,
                }
            };

            for child in &children {
                self.rigid_body_trickle_down_update(child, position, rotation);
            }

            if let Some(handle) = instance_handle {
                let scale = self.get_instance(handle).unwrap().transform.scale;

                self.update_instance(
                    handle,
                    Transform {
                        position,
                        rotation,
                        scale,
                    },
                );
            }

            match &mut self.entities[i] {
                EntityRef::DynamicBody(rigid_body)
                | EntityRef::StaticBody(rigid_body)
                | EntityRef::KinematicBody(rigid_body) => {
                    rigid_body.children = children;
                }
                _ => unreachable!(),
            }
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();
        if !self.is_surface_configured {
            return Ok(());
        }

        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            rp.set_pipeline(&self.render_pipeline);
            rp.set_bind_group(1, &self.camera_bind_group, &[]);

            for mesh_data in &self.meshes {
                let instance_count = mesh_data.instances.len() as u32;
                if instance_count == 0 {
                    continue; // Skip if no instances, Not really, or kinda?
                }

                let mesh = &mesh_data.mesh;

                rp.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                rp.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

                rp.set_vertex_buffer(1, mesh_data.instance_buffer.slice(..));

                rp.set_bind_group(0, &mesh_data.material.bind_group, &[]);

                rp.draw_indexed(0..mesh.index_count, 0, 0..instance_count);
            }
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();

        Ok(())
    }
}

pub struct Scene<'a> {
    core: &'a mut EngineState,
}

// User facing abstraction of EngineState

impl<'a> Scene<'a> {
    pub(crate) fn new(state: &'a mut EngineState) -> Self {
        Self { core: state }
    }

    pub fn get_entity(&mut self, entity_handle: EntityHandle) -> &mut EntityRef {
        self.core.get_entity(entity_handle)
    }

    pub fn spawn(&mut self, entity: impl Into<Entity>) -> EntityHandle {
        self.core.spawn(entity)
    }

    pub fn load_obj(&mut self, path: &str) -> anyhow::Result<Vec<MeshHandle>> {
        self.core.load_obj(path)
    }

    pub fn add_force(&mut self, entity_handle: EntityHandle, force: Vector3<f32>) {
        self.core.add_force(entity_handle, force);
    }

    pub fn get_linvel(&mut self, entity_handle: EntityHandle) -> Vector3<f32> {
        self.core.get_linvel(entity_handle)
    }

    pub fn set_linvel(&mut self, entity_handle: EntityHandle, velocity: Vector3<f32>) {
        self.core.set_linvel(entity_handle, velocity);
    }

    pub fn get_angvel(&mut self, entity_handle: EntityHandle) -> Vector3<f32> {
        self.core.get_angvel(entity_handle)
    }

    pub fn set_angvel(&mut self, entity_handle: EntityHandle, velocity: Vector3<f32>) {
        self.core.set_angvel(entity_handle, velocity);
    }

    pub fn set_enabled_rotations(
        &mut self,
        entity_handle: EntityHandle,
        enable_x: bool,
        enable_y: bool,
        enable_z: bool,
    ) {
        self.core
            .set_enabled_rotations(entity_handle, enable_x, enable_y, enable_z);
    }
    pub fn apply_impulse(&mut self, entity_handle: EntityHandle, impulse: Vector3<f32>) {
        self.core.apply_impulse(entity_handle, impulse);
    }

    pub fn grab_cursor(&mut self) {
        self.core.grab_cursor();
    }

    pub fn release_cursor(&mut self) {
        self.core.release_cursor();
    }
}

pub enum EngineEvent {
    Key {
        physical_key: PhysicalKey,
        pressed: bool,
    },
    MouseMotion {
        delta_x: f64,
        delta_y: f64,
    },
    MouseButton {
        button: winit::event::MouseButton,
        pressed: bool,
    },
}

pub trait Game {
    fn on_init(&mut self, _state: &mut Scene) {}
    fn on_update(&mut self, _delta_time: f32, _state: &mut Scene) {}
    fn on_event(&mut self, _event: EngineEvent, _state: &mut Scene) {}
}

struct App<'a, G: Game> {
    state: Option<EngineState>,
    game: &'a mut G,
    last_frame_time: std::time::Instant,
    width: u32,
    height: u32,
    title: String,
    fullscreen: bool,
    resizable: bool,
}

pub struct Runner<G: Game> {
    game: G,
    width: u32,
    height: u32,
    title: String,
    fullscreen: bool,
    resizable: bool,
}

impl<G: Game> Runner<G> {
    pub fn new(game: G) -> Self {
        Self {
            game,
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

    pub fn run(mut self) -> anyhow::Result<()> {
        env_logger::init();

        let event_loop = EventLoop::with_user_event().build()?;
        let mut app = App {
            state: None,
            game: &mut self.game,
            last_frame_time: std::time::Instant::now(),
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

// Based on the window "event" or action we run the correct function in game.

impl<'a, G: Game> ApplicationHandler<EngineState> for App<'a, G> {
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

                state.update();
                {
                    let mut scene = Scene::new(state);
                    self.game.on_update(delta, &mut scene);
                }

                match state.render() {
                    Ok(_) => {}
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
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key,
                        state: key_state,
                        ..
                    },
                ..
            } => {
                let mut scene = Scene::new(state);
                self.game.on_event(
                    EngineEvent::Key {
                        physical_key,
                        pressed: key_state.is_pressed(),
                    },
                    &mut scene,
                );
            }

            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => {
                let mut scene = Scene::new(state);
                self.game.on_event(
                    EngineEvent::MouseButton {
                        button,
                        pressed: button_state == ElementState::Pressed,
                    },
                    &mut scene,
                );
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

        // Handle raw mouse motion for unlimited camera rotation
        if let winit::event::DeviceEvent::MouseMotion { delta } = event {
            if state.cursor_grabbed {
                let mut scene = Scene::new(state);
                self.game.on_event(
                    EngineEvent::MouseMotion {
                        delta_x: delta.0,
                        delta_y: delta.1,
                    },
                    &mut scene,
                );
            }
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        use winit::window::Fullscreen;

        let mut window_attributes = Window::default_attributes()
            .with_title(&self.title)
            .with_inner_size(winit::dpi::PhysicalSize::new(self.width, self.height))
            .with_resizable(self.resizable);

        // Set fullscreen mode if requested
        if self.fullscreen {
            window_attributes =
                window_attributes.with_fullscreen(Some(Fullscreen::Borderless(None)));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        let mut state = pollster::block_on(EngineState::new(window.clone())).unwrap();

        {
            let mut scene = Scene::new(&mut state);
            self.game.on_init(&mut scene);
        }
        self.state = Some(state);
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: EngineState) {
        self.state = Some(event);
    }
}
