use crate::{
    entity::{
        CameraRef, DynamicBodyRef, EmptyRef, Entity, EntityBuilder, EntityHandle, EntityRef,
        KinematicBodyRef, MeshInstanceRef, OPENGL_TO_WGPU_MATRIX, PointLightRef, StaticBodyRef,
        get_entity_from_handle, spawn,
    },
    model::{InstanceRaw, MeshData, MeshHandle, load_obj},
    physics::PhysicsWorld,
};
use nalgebra::{
    self, Isometry, Matrix4, Perspective3, Quaternion, Translation3, UnitQuaternion, Vector3,
};
use rapier3d::prelude::ColliderHandle;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    sync::Arc,
};
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
const MAX_LIGHTS: usize = 100;

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
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    position: [f32; 3],
    _padding: f32,
    color: [f32; 3],
    _padding2: f32,
    strength: f32,
    _padding3: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct LightHandle(pub usize);

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
    pub physics_world: PhysicsWorld,
    entities: Vec<Entity>,
    collider_entity_pairs: HashMap<ColliderHandle, EntityHandle>,
    active_collisions: HashSet<(EntityHandle, EntityHandle)>,
    cursor_grabbed: bool,
    lights: Vec<LightUniform>,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: vertex_layouts,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState {
                    alpha: wgpu::BlendComponent::REPLACE,
                    color: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
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
            view_position: [0.0; 4],
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
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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

        // Create depth texture

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth-texture");

        // Lighting

        let light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light VB"),
            size: (std::mem::size_of::<LightUniform>() * MAX_LIGHTS) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: None,
            });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
            label: None,
        });

        // Create render pipeline

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                    &light_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Normal Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
            )
        };

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
            collider_entity_pairs: HashMap::new(),
            active_collisions: HashSet::new(),
            cursor_grabbed: false,
            lights: vec![],
            light_buffer,
            light_bind_group,
        })
    }

    // Loading models

    pub fn load_obj(&mut self, path: &str) -> anyhow::Result<Vec<MeshHandle>> {
        load_obj(
            &self.device,
            &self.queue,
            &self.texture_bind_group_layout,
            path,
            &mut self.meshes,
        )
    }

    // Entity management

    pub fn get_entity<'a>(
        &'a mut self,
        entity_handle: EntityHandle,
    ) -> anyhow::Result<EntityRef<'a>> {
        let entity = get_entity_from_handle(&mut self.entities, entity_handle)?;
        match entity {
            Entity::StaticBody(e) => Ok(EntityRef::StaticBody(StaticBodyRef { entity: e })),
            Entity::DynamicBody(e) => Ok(EntityRef::DynamicBody(DynamicBodyRef {
                entity: e,
                physics_world: &mut self.physics_world,
            })),
            Entity::KinematicBody(e) => Ok(EntityRef::KinematicBody(KinematicBodyRef {
                entity: e,
                physics_world: &mut self.physics_world,
            })),
            Entity::MeshInstance(e) => Ok(EntityRef::MeshInstance(MeshInstanceRef { entity: e })),
            Entity::Camera(e) => Ok(EntityRef::Camera(CameraRef { entity: e })),
            Entity::Empty(e) => Ok(EntityRef::Empty(EmptyRef { entity: e })),
            Entity::PointLight(e) => Ok(EntityRef::PointLight(PointLightRef { entity: e })),
        }
    }

    pub fn spawn(&mut self, entity: impl Into<EntityBuilder>) -> EntityHandle {
        spawn(
            &mut self.entities,
            &mut self.meshes,
            &mut self.collider_entity_pairs,
            &mut self.physics_world,
            &self.queue,
            &mut self.camera_uniform,
            &self.camera_buffer,
            &self.config,
            &mut self.lights,
            &self.light_buffer,
            entity,
        )
    }

    // Instances

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
        entity_ref: &Entity,
        parent_position: Vector3<f32>,
        parent_rotation: Quaternion<f32>,
    ) {
        // We want to update all non rigidbody children with the physics of the parent rigidbody.
        match &entity_ref {
            Entity::DynamicBody(entity) => {
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
            Entity::StaticBody(entity) => {
                let rigid_body_calc = self
                    .physics_world
                    .rigid_body_set
                    .get(entity.rigid_body_handle)
                    .unwrap();

                let iso = rigid_body_calc.position();
                let position: Vector3<f32> = iso.translation.vector;
                let rotation = iso.rotation.into_inner();

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
            Entity::KinematicBody(entity) => {
                let rigid_body_calc = self
                    .physics_world
                    .rigid_body_set
                    .get(entity.rigid_body_handle)
                    .unwrap();

                let iso = rigid_body_calc.position();
                let position: Vector3<f32> = iso.translation.vector;
                let rotation = iso.rotation.into_inner();

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
            Entity::MeshInstance(entity) => {
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
            Entity::Camera(entity) => {
                let rotated_offset = UnitQuaternion::from_quaternion(parent_rotation)
                    .transform_vector(&entity.transform.position);

                let camera_position = parent_position + rotated_offset;

                let iso = Isometry::from_parts(
                    Translation3::from(camera_position), // Use the rotated position
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
            Entity::Empty(entity) => {
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
            Entity::PointLight(entity) => {
                let rotated_offset = UnitQuaternion::from_quaternion(parent_rotation)
                    .transform_vector(&entity.transform.position);

                let new_position = parent_position + rotated_offset;

                let new_rotation = (UnitQuaternion::from_quaternion(parent_rotation)
                    * UnitQuaternion::from_quaternion(entity.transform.rotation))
                .into_inner();

                self.lights[entity.light_handle.0].position = new_position.into();
                self.lights[entity.light_handle.0].color = entity.color;
                self.lights[entity.light_handle.0].strength = entity.strenght;

                for child in &entity.children {
                    self.rigid_body_trickle_down_update(child, new_position, new_rotation);
                }
            }
        }
    }

    pub fn update(
        &mut self,
    ) -> (
        Vec<(EntityHandle, EntityHandle)>,
        Vec<(EntityHandle, EntityHandle)>,
    ) {
        self.physics_world.step();

        // This system is weird, it should be reworked.

        for i in 0..self.entities.len() {
            let (position, rotation, instance_handle, children) = {
                let entity_info = &mut self.entities[i];

                match entity_info {
                    Entity::DynamicBody(rigid_body) => {
                        let handle = rigid_body.rigid_body_handle;
                        let rigid_body_calc =
                            self.physics_world.rigid_body_set.get(handle).unwrap();

                        let iso = rigid_body_calc.position();
                        let position: Vector3<f32> = iso.translation.vector;
                        let rotation = iso.rotation.into_inner();

                        let children = std::mem::take(&mut rigid_body.children);

                        (position, rotation, rigid_body.instance_handle, children)
                    }
                    Entity::StaticBody(rigid_body) => {
                        let handle = rigid_body.rigid_body_handle;
                        let rigid_body_calc =
                            self.physics_world.rigid_body_set.get(handle).unwrap();

                        let iso = rigid_body_calc.position();
                        let position: Vector3<f32> = iso.translation.vector;
                        let rotation = iso.rotation.into_inner();

                        let children = std::mem::take(&mut rigid_body.children);

                        (position, rotation, rigid_body.instance_handle, children)
                    }
                    Entity::KinematicBody(rigid_body) => {
                        let handle = rigid_body.rigid_body_handle;
                        let rigid_body_calc =
                            self.physics_world.rigid_body_set.get(handle).unwrap();

                        let iso = rigid_body_calc.position();
                        let position: Vector3<f32> = iso.translation.vector;
                        let rotation = iso.rotation.into_inner();

                        let children = std::mem::take(&mut rigid_body.children);

                        (position, rotation, rigid_body.instance_handle, children)
                    }
                    Entity::MeshInstance(mesh_instance) => {
                        let children = std::mem::take(&mut mesh_instance.children);
                        (
                            mesh_instance.transform.position,
                            mesh_instance.transform.rotation,
                            Some(mesh_instance.instance_handle),
                            children,
                        )
                    }
                    Entity::Empty(empty) => {
                        let children = std::mem::take(&mut empty.children);
                        (
                            empty.transform.position,
                            empty.transform.rotation,
                            None,
                            children,
                        )
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
                Entity::DynamicBody(rigid_body) => {
                    rigid_body.children = children;
                }
                Entity::StaticBody(rigid_body) => {
                    rigid_body.children = children;
                }
                Entity::KinematicBody(rigid_body) => {
                    rigid_body.children = children;
                }
                Entity::MeshInstance(mesh_instance) => {
                    mesh_instance.children = children;
                }
                Entity::Empty(empty) => {
                    empty.children = children;
                }
                _ => unreachable!(),
            }
        }

        // Lights

        self.queue
            .write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&self.lights));

        // Collisions

        let current_collisions: HashSet<(EntityHandle, EntityHandle)> = self
            .physics_world
            .narrow_phase
            .contact_pairs()
            .map(|contact_pair| {
                let handle1 = self
                    .collider_entity_pairs
                    .get(&contact_pair.collider1)
                    .unwrap()
                    .clone();
                let handle2 = self
                    .collider_entity_pairs
                    .get(&contact_pair.collider2)
                    .unwrap()
                    .clone();

                // Ensure that it's the same no matter what collider is 1 or 2.
                if handle1 <= handle2 {
                    (handle1, handle2)
                } else {
                    (handle2, handle1)
                }
            })
            .collect();

        let new_collisions: Vec<_> = current_collisions
            .difference(&self.active_collisions)
            .cloned()
            .collect();

        let removed_collisions: Vec<_> = self
            .active_collisions
            .difference(&current_collisions)
            .cloned()
            .collect();

        self.active_collisions = current_collisions;

        (new_collisions, removed_collisions)
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

            // Meshes

            rp.set_pipeline(&self.render_pipeline);
            rp.set_bind_group(1, &self.camera_bind_group, &[]);
            rp.set_bind_group(2, &self.light_bind_group, &[]);

            for mesh_data in &self.meshes {
                let instance_count = mesh_data.instances.len() as u32;
                if instance_count == 0 {
                    continue; // Skip if no instances
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

pub struct SceneInstance {
    pub scene: Box<dyn Scene>,
    pub is_active: bool,
}

pub struct SceneContext<'a> {
    core: &'a mut EngineState,
    scenes: &'a mut Vec<SceneInstance>,
}

// User facing abstraction of EngineState

impl<'a> SceneContext<'a> {
    pub(crate) fn new(state: &'a mut EngineState, scenes: &'a mut Vec<SceneInstance>) -> Self {
        Self {
            core: state,
            scenes,
        }
    }

    pub fn get_entity(&mut self, entity_handle: EntityHandle) -> anyhow::Result<EntityRef<'_>> {
        self.core.get_entity(entity_handle)
    }

    pub fn spawn(&mut self, entity: impl Into<EntityBuilder>) -> EntityHandle {
        self.core.spawn(entity)
    }

    pub fn spawn_scene(&mut self, scene: impl Scene + 'static) {
        self.scenes.push(SceneInstance {
            scene: Box::new(scene),
            is_active: false,
        });
    }

    pub fn get_scene() {
        // Allows you to set the transform (scale, position, rotation)
    }

    pub fn load_obj(&mut self, path: &str) -> anyhow::Result<Vec<MeshHandle>> {
        self.core.load_obj(path)
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
    CollisionEnter {
        entity: EntityHandle,
        other: EntityHandle,
    },
    CollisionExit {
        entity: EntityHandle,
        other: EntityHandle,
    },
}

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

pub trait Scene {
    fn on_init(&mut self, _ctx: &mut SceneContext) {}
    fn on_update(&mut self, _delta_time: f32, _ctx: &mut SceneContext) {}
    fn on_event(&mut self, _event: EngineEvent, _ctx: &mut SceneContext) {}
}

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
