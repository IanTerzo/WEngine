use bytemuck::{Pod, Zeroable};
use cgmath::{Quaternion, Vector3, prelude::*};
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::{event_loop::ActiveEventLoop, keyboard::KeyCode, window::Window};

mod texture;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
}

impl Vertex {
    const ATTRS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as _,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::from_cols(
    cgmath::Vector4::new(1.0, 0.0, 0.0, 0.0),
    cgmath::Vector4::new(0.0, 1.0, 0.0, 0.0),
    cgmath::Vector4::new(0.0, 0.0, 0.5, 0.0),
    cgmath::Vector4::new(0.0, 0.0, 0.5, 1.0),
);

struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        OPENGL_TO_WGPU_MATRIX * proj * view
    }
}

pub struct Transform {
    pub position: Vector3<f32>,
    pub rotation: Quaternion<f32>,
    pub scale: Vector3<f32>,
}

impl Transform {
    pub fn to_matrix(&self) -> cgmath::Matrix4<f32> {
        let translation = cgmath::Matrix4::from_translation(self.position);
        let rotation = cgmath::Matrix4::from(self.rotation);
        let scale =
            cgmath::Matrix4::from_nonuniform_scale(self.scale.x, self.scale.y, self.scale.z);
        translation * rotation * scale
    }
}

pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

pub struct Material {
    pub bind_group: wgpu::BindGroup,
}

pub struct MeshData {
    pub mesh: Mesh,
    pub material: Material,
    pub instance_buffer: wgpu::Buffer,
    pub instances: Vec<Instance>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeshHandle(usize);

pub struct Instance {
    pub transform: Transform,
}

impl<'a> Instance {
    pub fn to_raw(&self) -> InstanceRaw {
        let model_matrix = self.transform.to_matrix(); // convert Transform to 4x4 matrix
        InstanceRaw {
            model_matrix: model_matrix.into(),
            // optional:
            // material_index: self.material.id as u32,
            // _padding: [0; 3],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct InstanceRaw {
    pub model_matrix: [[f32; 4]; 4], // 4x4 transform matrix
}

impl InstanceRaw {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance, // Important: advances per instance
            attributes: &[
                // model_matrix is 4 vec4's, one for each row
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5, // matches your WGSL shader
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new(camera: &Camera) -> Self {
        Self {
            view_proj: camera.build_view_projection_matrix().into(),
        }
    }
}

pub struct State {
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
    materials: Vec<Material>,
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<State> {
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

        let camera = Camera {
            eye: (0.0, 3.0, 5.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let camera_uniform = CameraUniform::new(&camera);

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
                buffers: &[Vertex::desc(), InstanceRaw::desc()],
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
            materials: vec![],
        })
    }

    pub fn load_material(&mut self) -> Material {
        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_texture = texture::Texture::from_bytes(
            &self.device,
            &self.queue,
            diffuse_bytes,
            "happy-tree.png",
        )
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
            label: Some("diffuse_bind_group"),
        });

        Material { bind_group }
    }
    pub fn add_mesh(
        &mut self,
        vertices: &[Vertex],
        indices: &[u16],
        material: Material,
    ) -> MeshHandle {
        let vertex_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        const MAX_INSTANCES: usize = 100;

        let instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (std::mem::size_of::<InstanceRaw>() * MAX_INSTANCES) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mesh_data = MeshData {
            mesh: Mesh {
                vertex_buffer,
                index_buffer,
                index_count: indices.len() as u32,
            },
            material,
            instance_buffer,
            instances: vec![],
        };

        self.meshes.push(mesh_data);

        MeshHandle(self.meshes.len() - 1)
    }

    pub fn add_instance(&mut self, handle: MeshHandle, transform: Transform) -> &Instance {
        if let Some(mesh_data) = self.meshes.get_mut(handle.0) {
            mesh_data.instances.push(Instance { transform });
            let instance_ref = mesh_data.instances.last().unwrap();

            let instance_raw = instance_ref.to_raw();
            let offset = (mesh_data.instances.len() - 1) * std::mem::size_of::<InstanceRaw>();

            self.queue.write_buffer(
                &mesh_data.instance_buffer,
                offset as wgpu::BufferAddress,
                bytemuck::cast_slice(&[instance_raw]),
            );

            instance_ref
        } else {
            panic!("Couldn't not find mesh by handle");
        }
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

    pub fn handle_key(&self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        if (code, is_pressed) == (KeyCode::Escape, true) {
            event_loop.exit();
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
                let mesh = &mesh_data.mesh;

                let instance_count = mesh_data.instances.len() as u32;
                if instance_count == 0 {
                    continue; // Skip if no instances
                }

                rp.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                rp.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

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
        // remove `todo!()`
    }
}
