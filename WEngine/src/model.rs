use bytemuck::{Pod, Zeroable};
use nalgebra::{Matrix4, Quaternion, Translation3, UnitQuaternion, Vector3};

use crate::texture;

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

pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Material {
    #[allow(unused)]
    pub name: String,
    #[allow(unused)]
    pub diffuse_texture: texture::Texture,
    pub bind_group: wgpu::BindGroup,
}

pub struct MeshData {
    pub mesh: Mesh,
    pub material: Material,
    pub instance_buffer: wgpu::Buffer,
    pub instances: Vec<Instance>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct InstanceHandle {
    pub mesh: MeshHandle,
    pub instance_index: usize,
}

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
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
}

impl ModelVertex {
    const ATTRS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ModelVertex>() as _,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}
