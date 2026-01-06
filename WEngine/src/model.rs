use crate::{Instance, MAX_INSTANCES, texture};
use bytemuck::{Pod, Zeroable};
use std::{fs::File, io::BufReader, path::Path};
use wgpu::util::DeviceExt;

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

pub fn load_obj(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    path: &str,
    meshes: &mut Vec<MeshData>,
) -> anyhow::Result<Vec<MeshHandle>> {
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
                &device,
                &queue,
                path.parent()
                    .unwrap()
                    .join(&mat.diffuse_texture)
                    .to_str()
                    .unwrap(),
            )?;
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
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
            texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "error.png").unwrap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
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

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mesh_struct = Mesh {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as u32,
        };

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
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

        meshes.push(MeshData {
            mesh: mesh_struct,
            material,
            instance_buffer,
            instances: Vec::new(),
        });

        mesh_handle_list.push(MeshHandle(meshes.len() - 1));
    }

    Ok(mesh_handle_list)
}
