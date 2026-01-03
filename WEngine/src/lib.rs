use crate::{
    model::{
        Instance, InstanceHandle, InstanceRaw, Material, Mesh, MeshData, MeshHandle, ModelVertex,
        Transform,
    },
    physics::{Body, ColliderConfig, DynamicBody, KinematicBody, PhysicsWorld, StaticBody},
};
use nalgebra::{self, Matrix4, Perspective3, Point3, UnitQuaternion, Vector3};
use rapier3d::prelude::{ColliderBuilder, RigidBodyHandle};
use std::{fs::File, io::BufReader, path::Path, sync::Arc};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

pub mod model;
pub mod physics;
pub mod texture;

const MAX_INSTANCES: usize = 100;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

pub struct CameraInfo {
    pub eye: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Default for CameraInfo {
    fn default() -> Self {
        Self {
            eye: Point3::new(0.0, 3.0, 5.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::y(),
            aspect: 16.0 / 9.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        }
    }
}

impl CameraInfo {
    pub fn build_view_projection_matrix(&self) -> Matrix4<f32> {
        let view = Matrix4::look_at_rh(&self.eye, &self.target, &self.up);

        let proj = Perspective3::new(self.aspect, self.fovy.to_radians(), self.znear, self.zfar)
            .to_homogeneous();

        OPENGL_TO_WGPU_MATRIX * proj * view
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &CameraInfo) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

pub struct RigidBodyData {
    instance_handle: InstanceHandle,
    rigid_body_handle: RigidBodyHandle,
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
    rigid_bodies: Vec<RigidBodyData>,
    last_mouse_position: Option<PhysicalPosition<f64>>,
    cursor_grabbed: bool,
}

impl EngineState {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<EngineState> {
        // Create the instance and config

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

        // Get device and queue

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await?;

        // Create camera

        let camera_uniform = CameraUniform::new();

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
            rigid_bodies: vec![],
            last_mouse_position: None,
            cursor_grabbed: false,
        })
    }

    pub fn load_obj(&mut self, path: &str) -> anyhow::Result<Vec<MeshHandle>> {
        let path = Path::new(path);
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Load OBJ file synchronously
        let (models, materials) = tobj::load_obj_buf(
            &mut reader,
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
            |mat_path| {
                let mat_file = File::open(path.parent().unwrap().join(mat_path)).unwrap();
                let mut mat_reader = BufReader::new(mat_file);
                Ok(tobj::load_mtl_buf(&mut mat_reader)?)
            },
        )?;

        // Load WGPU materials
        let mut wgpu_materials = Vec::new();
        if materials.clone()?.len() > 0 {
            for mat in materials? {
                let diffuse_texture = texture::Texture::from_file(
                    &self.device,
                    &self.queue,
                    path.parent()
                        .unwrap()
                        .join(&mat.diffuse_texture)
                        .to_str()
                        .unwrap(),
                )?;
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                        },
                    ],
                    label: None,
                });

                wgpu_materials.push(Material {
                    name: mat.name,
                    diffuse_texture,
                    bind_group,
                });
            }
        } else {
            let diffuse_bytes = include_bytes!("error.png");
            let diffuse_texture =
                texture::Texture::from_bytes(&self.device, &self.queue, diffuse_bytes, "error.png")
                    .unwrap();

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                    },
                ],
                label: None,
            });

            wgpu_materials.push(Material {
                name: "default".to_string(),
                diffuse_texture: diffuse_texture,
                bind_group: bind_group,
            })
        }

        // Convert OBJ meshes into MeshData
        let mut mesh_handle_list = Vec::new();
        for m in models {
            let mesh = &m.mesh;

            let vertices: Vec<ModelVertex> = (0..mesh.positions.len() / 3)
                .map(|i| ModelVertex {
                    position: [
                        mesh.positions[i * 3],
                        mesh.positions[i * 3 + 1],
                        mesh.positions[i * 3 + 2],
                    ],
                    tex_coords: if !mesh.texcoords.is_empty() {
                        [mesh.texcoords[i * 2], 1.0 - mesh.texcoords[i * 2 + 1]]
                    } else {
                        [0.0, 0.0]
                    },
                    normal: if !mesh.normals.is_empty() {
                        [
                            mesh.normals[i * 3],
                            mesh.normals[i * 3 + 1],
                            mesh.normals[i * 3 + 2],
                        ]
                    } else {
                        [0.0, 0.0, 0.0]
                    },
                })
                .collect();

            let vertex_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

            let index_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Index Buffer"),
                    contents: bytemuck::cast_slice(&mesh.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

            let mesh_struct = Mesh {
                vertex_buffer,
                index_buffer,
                index_count: mesh.indices.len() as u32,
            };

            // Create a default instance buffer (empty for now)
            let instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Instance Buffer"),
                size: (std::mem::size_of::<InstanceRaw>() * MAX_INSTANCES) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let material = if let Some(mat_id) = mesh.material_id {
                wgpu_materials[mat_id].clone()
            } else {
                wgpu_materials[0].clone() // fallback to first material
            };

            self.meshes.push(MeshData {
                mesh: mesh_struct,
                material,
                instance_buffer,
                instances: Vec::new(),
            });

            mesh_handle_list.push(MeshHandle(self.meshes.len() - 1));
        }

        Ok(mesh_handle_list)
    }

    pub fn instantiate(&mut self, handle: MeshHandle, transform: Transform) -> InstanceHandle {
        if let Some(mesh_data) = self.meshes.get_mut(handle.0) {
            mesh_data.instances.push(Instance { transform });
            let instance_index = mesh_data.instances.len() - 1;

            let instance_raw = mesh_data.instances.last().unwrap().to_raw();
            let offset = (mesh_data.instances.len() - 1) * std::mem::size_of::<InstanceRaw>();

            self.queue.write_buffer(
                &mesh_data.instance_buffer,
                offset as wgpu::BufferAddress,
                bytemuck::cast_slice(&[instance_raw]),
            );

            InstanceHandle {
                mesh: handle,
                instance_index: instance_index,
            }
        } else {
            panic!("Couldn't not find mesh by handle");
        }
    }

    pub fn spawn(&mut self, body: impl Into<Body>) -> RigidBodyHandle {
        let body = body.into();

        match body {
            Body::Dynamic(dynamic) => self.spawn_dynamic(dynamic),
            Body::Static(static_body) => self.spawn_static(static_body),
            Body::Kinematic(kinematic) => self.spawn_kinematic(kinematic),
        }
    }

    fn spawn_dynamic(&mut self, body: DynamicBody) -> RigidBodyHandle {
        let unit_quat = UnitQuaternion::from_quaternion(body.transform.rotation);
        let axis_angle = if let Some((axis, angle)) = unit_quat.axis_angle() {
            axis.into_inner() * angle
        } else {
            Vector3::zeros()
        };

        let rigid_body = rapier3d::prelude::RigidBodyBuilder::dynamic()
            .translation(body.transform.position)
            .rotation(axis_angle)
            .linvel(body.linear_velocity)
            .angvel(body.angular_velocity)
            .additional_mass(body.mass)
            .linear_damping(body.linear_damping)
            .angular_damping(body.angular_damping)
            .gravity_scale(body.gravity_scale)
            .can_sleep(body.can_sleep)
            .build();

        self.insert_body_and_collider(body.mesh_handle, body.transform, body.collider, rigid_body)
    }

    fn spawn_static(&mut self, body: StaticBody) -> RigidBodyHandle {
        let unit_quat = UnitQuaternion::from_quaternion(body.transform.rotation);
        let axis_angle = if let Some((axis, angle)) = unit_quat.axis_angle() {
            axis.into_inner() * angle
        } else {
            Vector3::zeros()
        };

        let rigid_body = rapier3d::prelude::RigidBodyBuilder::fixed()
            .translation(body.transform.position)
            .rotation(axis_angle)
            .build();

        self.insert_body_and_collider(body.mesh_handle, body.transform, body.collider, rigid_body)
    }

    fn spawn_kinematic(&mut self, body: KinematicBody) -> RigidBodyHandle {
        let unit_quat = UnitQuaternion::from_quaternion(body.transform.rotation);
        let axis_angle = if let Some((axis, angle)) = unit_quat.axis_angle() {
            axis.into_inner() * angle
        } else {
            Vector3::zeros()
        };

        let rigid_body = rapier3d::prelude::RigidBodyBuilder::kinematic_position_based()
            .translation(body.transform.position)
            .rotation(axis_angle)
            .linvel(body.linear_velocity)
            .angvel(body.angular_velocity)
            .build();

        self.insert_body_and_collider(body.mesh_handle, body.transform, body.collider, rigid_body)
    }

    fn insert_body_and_collider(
        &mut self,
        mesh_handle: MeshHandle,
        transform: Transform,
        collider_config: ColliderConfig,
        rigid_body: rapier3d::prelude::RigidBody,
    ) -> RigidBodyHandle {
        let mesh_data = self.meshes.get_mut(mesh_handle.0).unwrap();
        mesh_data.instances.push(Instance { transform });

        let rigid_body_handle = self.physics_world.rigid_body_set.insert(rigid_body);

        let collider = match collider_config {
            ColliderConfig::Ball { radius } => ColliderBuilder::ball(radius).build(),
            ColliderConfig::Capsule {
                half_height,
                radius,
            } => ColliderBuilder::capsule_y(half_height, radius).build(),
            ColliderConfig::Cuboid { half_extents } => {
                ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z).build()
            }
            ColliderConfig::Cylinder {
                half_height,
                radius,
            } => ColliderBuilder::cylinder(half_height, radius).build(),
            ColliderConfig::Custom(collider) => collider,
        };

        self.physics_world.collider_set.insert_with_parent(
            collider,
            rigid_body_handle,
            &mut self.physics_world.rigid_body_set,
        );

        let instance_index = mesh_data.instances.len() - 1;
        self.rigid_bodies.push(RigidBodyData {
            instance_handle: InstanceHandle {
                mesh: mesh_handle,
                instance_index,
            },
            rigid_body_handle,
        });

        rigid_body_handle
    }

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

    pub fn update_camera(&mut self, camera: &CameraInfo) {
        self.camera_uniform.update_view_proj(camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }
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

    pub fn update(&mut self) {
        self.physics_world.step();

        for i in 0..self.rigid_bodies.len() {
            let rigid_body = &self.rigid_bodies[i];

            let rigid_body_calc = self
                .physics_world
                .rigid_body_set
                .get(rigid_body.rigid_body_handle)
                .unwrap();

            let iso = rigid_body_calc.position();
            let position: Vector3<f32> = iso.translation.vector;
            let rotation = iso.rotation.into_inner();

            let scale = self
                .get_instance(rigid_body.instance_handle)
                .unwrap()
                .transform
                .scale;

            self.update_instance(
                rigid_body.instance_handle,
                Transform {
                    position,
                    rotation,
                    scale,
                },
            );
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
}

pub struct Scene<'a> {
    core: &'a mut EngineState,
}

impl<'a> Scene<'a> {
    pub(crate) fn new(state: &'a mut EngineState) -> Self {
        Self { core: state }
    }

    pub fn update_camera(&mut self, camera: &CameraInfo) {
        self.core.update_camera(camera);
    }

    pub fn load_obj(&mut self, path: &str) -> anyhow::Result<Vec<MeshHandle>> {
        self.core.load_obj(path)
    }

    pub fn instantiate(&mut self, handle: MeshHandle, transform: Transform) -> InstanceHandle {
        self.core.instantiate(handle, transform)
    }

    pub fn get_instance(&self, handle: InstanceHandle) -> Option<&Instance> {
        self.core.get_instance(handle)
    }

    pub fn update_instance(&mut self, handle: InstanceHandle, transform: Transform) {
        self.core.update_instance(handle, transform);
    }

    pub fn spawn(&mut self, body: impl Into<Body>) {
        self.core.spawn(body);
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

pub struct Runner<G: Game> {
    game: G,
}

struct App<'a, G: Game> {
    state: Option<EngineState>,
    game: &'a mut G,
    last_frame_time: std::time::Instant,
}

impl<G: Game> Runner<G> {
    pub fn run(game: G) -> anyhow::Result<()> {
        let mut runner = Self { game };
        runner.start()
    }

    fn start(&mut self) -> anyhow::Result<()> {
        env_logger::init();

        let event_loop = EventLoop::with_user_event().build()?;
        let mut app = App {
            state: None,
            game: &mut self.game,
            last_frame_time: std::time::Instant::now(),
        };

        event_loop.run_app(&mut app)?;

        Ok(())
    }
}

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
            WindowEvent::CursorMoved { position, .. } => {
                if state.cursor_grabbed {
                    if let Some(last_pos) = state.last_mouse_position {
                        let delta_x = position.x - last_pos.x;
                        let delta_y = position.y - last_pos.y;

                        let mut scene = Scene::new(state);
                        self.game
                            .on_event(EngineEvent::MouseMotion { delta_x, delta_y }, &mut scene);
                    }
                    state.last_mouse_position = Some(position);
                }
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

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();

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
