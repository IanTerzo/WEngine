mod pipeline;
use std::sync::Arc;

use nalgebra::Matrix4;
pub use pipeline::create_render_pipeline;
use winit::window::Window;

use crate::{
    camera::camera_bind_group_layout,
    instance::InstanceRaw,
    lightning::light_bind_group_layout,
    mesh::{MeshData, ModelVertex},
    texture,
};

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    standard_pipeline: wgpu::RenderPipeline,
    light_pipeline: wgpu::RenderPipeline,
    depth_texture: texture::Texture,
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub light_bind_group_layout: wgpu::BindGroupLayout,
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
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

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await?;

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

        let camera_bind_group_layout = camera_bind_group_layout(&device);

        let light_bind_group_layout = light_bind_group_layout(&device);

        let standard_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Standard Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                    &light_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Standard Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../shaders/standard.wgsl").into(),
                ),
            };
            create_render_pipeline(
                &device,
                &layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[ModelVertex::desc(), InstanceRaw::desc()],
                shader,
            )
        };

        let light_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });

            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Light Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/light.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[ModelVertex::desc(), InstanceRaw::desc()],
                shader,
            )
        };

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth-texture");

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            standard_pipeline,
            light_pipeline,
            depth_texture,
            texture_bind_group_layout,
            camera_bind_group_layout,
            light_bind_group_layout,
        })
    }

    pub fn render(
        &self,
        meshes: &[MeshData],
        camera_bind_group: &wgpu::BindGroup,
        light_bind_group: &wgpu::BindGroup,
    ) -> Result<(), wgpu::SurfaceError> {
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

            for mesh_data in meshes {
                let mesh = &mesh_data.mesh;

                let standarde_instance_count = mesh_data.standard_instances.len() as u32;
                if standarde_instance_count != 0 {
                    rp.set_pipeline(&self.standard_pipeline);
                    rp.set_bind_group(1, camera_bind_group, &[]);
                    rp.set_bind_group(2, light_bind_group, &[]);

                    rp.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    rp.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

                    rp.set_vertex_buffer(1, mesh_data.standard_instance_buffer.slice(..));

                    rp.set_bind_group(0, &mesh_data.material.bind_group, &[]);

                    rp.draw_indexed(0..mesh.index_count, 0, 0..standarde_instance_count);
                }

                let light_instance_count = mesh_data.light_instances.len() as u32;
                if light_instance_count != 0 {
                    rp.set_pipeline(&self.light_pipeline);
                    rp.set_bind_group(1, camera_bind_group, &[]);

                    rp.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    rp.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

                    rp.set_vertex_buffer(1, mesh_data.light_instance_buffer.slice(..));

                    rp.set_bind_group(0, &mesh_data.material.bind_group, &[]);

                    rp.draw_indexed(0..mesh.index_count, 0, 0..light_instance_count);
                }
            }
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();

        Ok(())
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

    pub fn flush_meshes(&self, meshes: &[MeshData]) {
        for mesh_data in meshes {
            if !mesh_data.standard_instances.is_empty() {
                let raw: Vec<InstanceRaw> = mesh_data
                    .standard_instances
                    .iter()
                    .map(|i| i.to_raw())
                    .collect();
                self.queue.write_buffer(
                    &mesh_data.standard_instance_buffer,
                    0,
                    bytemuck::cast_slice(&raw),
                );
            }
            if !mesh_data.light_instances.is_empty() {
                let raw: Vec<InstanceRaw> = mesh_data
                    .light_instances
                    .iter()
                    .map(|i| i.to_raw())
                    .collect();
                self.queue.write_buffer(
                    &mesh_data.light_instance_buffer,
                    0,
                    bytemuck::cast_slice(&raw),
                );
            }
        }
    }
}
