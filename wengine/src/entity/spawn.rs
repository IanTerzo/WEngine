use nalgebra::{Isometry3, Perspective3, Translation, UnitQuaternion, Vector3};
use rapier3d::prelude::{Collider, ColliderBuilder, ColliderHandle, RigidBodyHandle};
use std::collections::HashMap;

use crate::{
    camera::CameraState,
    entity::{
        Camera, DynamicBody, Empty, Entity, EntityHandle, KinematicBody, MeshInstance, PointLight,
        StaticBody,
        builder::{
            CameraBuilder, DynamicBodyBuilder, EmptyBuilder, EntityBuilder, KinematicBodyBuilder,
            MeshInstanceBuilder, PointLightBuilder, StaticBodyBuilder,
        },
    },
    instance::{Instance, InstanceHandle, InstanceType},
    lightning::{LightHandle, LightUniform, LightingState},
    mesh::{MeshData, MeshHandle},
    physics::{ColliderConfig, PhysicsWorld},
    renderer::OPENGL_TO_WGPU_MATRIX,
    transform::Transform,
};

pub struct SpawnContext<'a> {
    pub entities: &'a mut Vec<Entity>,
    pub meshes: &'a mut Vec<MeshData>,
    pub collider_entity_pairs: &'a mut HashMap<ColliderHandle, EntityHandle>,
    pub physics_world: &'a mut PhysicsWorld,
    pub camera: &'a mut CameraState,
    pub lighting: &'a mut LightingState,
    pub queue: &'a wgpu::Queue,
    pub config: &'a wgpu::SurfaceConfiguration,
}

impl<'a> SpawnContext<'a> {
    pub fn spawn(&mut self, entity: impl Into<EntityBuilder>) -> EntityHandle {
        let root = self.entities.len();
        let entity = self.create(root, vec![], entity);
        self.entities.push(entity);
        EntityHandle { root, path: vec![] }
    }

    fn create(
        &mut self,
        root: usize,
        path: Vec<usize>,
        entity: impl Into<EntityBuilder>,
    ) -> Entity {
        match entity.into() {
            EntityBuilder::DynamicBody(b) => self.create_dynamic(root, path, b),
            EntityBuilder::StaticBody(b) => self.create_static(root, path, b),
            EntityBuilder::KinematicBody(b) => self.create_kinematic(root, path, b),
            EntityBuilder::MeshInstance(b) => self.create_mesh_instance(root, path, b),
            EntityBuilder::Camera(b) => self.create_camera(b),
            EntityBuilder::Empty(b) => self.create_empty(root, path, b),
            EntityBuilder::PointLight(b) => self.create_point_light(root, path, b),
        }
    }

    fn create_children(
        &mut self,
        root: usize,
        path: &[usize],
        children: Vec<EntityBuilder>,
    ) -> Vec<Entity> {
        children
            .into_iter()
            .enumerate()
            .map(|(i, child)| {
                let mut child_path = path.to_vec();
                child_path.push(i);
                self.create(root, child_path, child)
            })
            .collect()
    }

    fn build_collider(config: ColliderConfig) -> Collider {
        match config {
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
        }
    }

    fn insert_collider(
        &mut self,
        config: Option<ColliderConfig>,
        rigid_body_handle: RigidBodyHandle,
        entity_handle: EntityHandle,
    ) {
        if let Some(config) = config {
            let collider = Self::build_collider(config);
            let collider_handle = self.physics_world.collider_set.insert_with_parent(
                collider,
                rigid_body_handle,
                &mut self.physics_world.rigid_body_set,
            );
            self.collider_entity_pairs
                .insert(collider_handle, entity_handle);
        }
    }

    fn push_standard_instance(
        &mut self,
        mesh_handle: MeshHandle,
        transform: Transform,
    ) -> InstanceHandle {
        let mesh_data = self.meshes.get_mut(mesh_handle.0).unwrap();
        mesh_data.standard_instances.push(Instance { transform });
        let instance_index = mesh_data.standard_instances.len() - 1;
        InstanceHandle {
            mesh: mesh_handle,
            instance_index,
            instance_type: InstanceType::Standard,
        }
    }

    fn to_axis_angle(rotation: nalgebra::Quaternion<f32>) -> Vector3<f32> {
        let unit = UnitQuaternion::from_quaternion(rotation);
        unit.axis_angle()
            .map(|(axis, angle)| axis.into_inner() * angle)
            .unwrap_or_else(Vector3::zeros)
    }

    fn create_dynamic(
        &mut self,
        root: usize,
        path: Vec<usize>,
        body: DynamicBodyBuilder,
    ) -> Entity {
        let rigid_body = rapier3d::prelude::RigidBodyBuilder::dynamic()
            .translation(body.transform.position)
            .rotation(Self::to_axis_angle(body.transform.rotation))
            .linvel(body.linear_velocity)
            .angvel(body.angular_velocity)
            .additional_mass(body.mass)
            .linear_damping(body.linear_damping)
            .angular_damping(body.angular_damping)
            .gravity_scale(body.gravity_scale)
            .can_sleep(body.can_sleep)
            .build();

        let instance_handle = body
            .mesh_handle
            .map(|h| self.push_standard_instance(h, body.transform));

        let rigid_body_handle = self.physics_world.rigid_body_set.insert(rigid_body);

        self.insert_collider(
            body.collider,
            rigid_body_handle,
            EntityHandle {
                root,
                path: path.clone(),
            },
        );

        let children = self.create_children(root, &path, body.children);

        Entity::DynamicBody(DynamicBody {
            tag: body.tag,
            transform: body.transform,
            instance_handle,
            rigid_body_handle,
            children,
        })
    }

    fn create_static(&mut self, root: usize, path: Vec<usize>, body: StaticBodyBuilder) -> Entity {
        let rigid_body = rapier3d::prelude::RigidBodyBuilder::fixed()
            .translation(body.transform.position)
            .rotation(Self::to_axis_angle(body.transform.rotation))
            .build();

        let instance_handle = body
            .mesh_handle
            .map(|h| self.push_standard_instance(h, body.transform));

        let rigid_body_handle = self.physics_world.rigid_body_set.insert(rigid_body);

        self.insert_collider(
            body.collider,
            rigid_body_handle,
            EntityHandle {
                root,
                path: path.clone(),
            },
        );

        let children = self.create_children(root, &path, body.children);

        Entity::StaticBody(StaticBody {
            tag: body.tag,
            transform: body.transform,
            instance_handle,
            rigid_body_handle,
            children,
        })
    }

    fn create_kinematic(
        &mut self,
        root: usize,
        path: Vec<usize>,
        body: KinematicBodyBuilder,
    ) -> Entity {
        let rigid_body = rapier3d::prelude::RigidBodyBuilder::kinematic_position_based()
            .translation(body.transform.position)
            .rotation(Self::to_axis_angle(body.transform.rotation))
            .linvel(body.linear_velocity)
            .angvel(body.angular_velocity)
            .build();

        let instance_handle = body
            .mesh_handle
            .map(|h| self.push_standard_instance(h, body.transform));

        let rigid_body_handle = self.physics_world.rigid_body_set.insert(rigid_body);

        self.insert_collider(
            body.collider,
            rigid_body_handle,
            EntityHandle {
                root,
                path: path.clone(),
            },
        );

        let children = self.create_children(root, &path, body.children);

        Entity::KinematicBody(KinematicBody {
            tag: body.tag,
            transform: body.transform,
            instance_handle,
            rigid_body_handle,
            children,
        })
    }

    fn create_mesh_instance(
        &mut self,
        root: usize,
        path: Vec<usize>,
        b: MeshInstanceBuilder,
    ) -> Entity {
        let instance_handle = self.push_standard_instance(b.mesh_handle, b.transform);
        let children = self.create_children(root, &path, b.children);

        Entity::MeshInstance(MeshInstance {
            tag: b.tag,
            instance_handle,
            transform: b.transform,
            children,
        })
    }

    fn create_camera(&mut self, camera: CameraBuilder) -> Entity {
        let iso = Isometry3::from_parts(
            Translation::from(camera.transform.position),
            UnitQuaternion::from_quaternion(camera.transform.rotation),
        );
        let view = iso.inverse().to_homogeneous();
        let aspect = self.config.width as f32 / self.config.height as f32;
        let proj = Perspective3::new(aspect, camera.fov.to_radians(), camera.near, camera.far)
            .to_homogeneous();

        self.camera.uniform.view_position = camera.transform.position.to_homogeneous().into();
        self.camera
            .update_view_proj(self.queue, OPENGL_TO_WGPU_MATRIX * proj * view);

        Entity::Camera(Camera {
            tag: camera.tag,
            transform: camera.transform,
            fov: camera.fov,
            near: camera.near,
            far: camera.far,
        })
    }

    fn create_empty(&mut self, root: usize, path: Vec<usize>, empty: EmptyBuilder) -> Entity {
        let children = self.create_children(root, &path, empty.children);

        Entity::Empty(Empty {
            tag: empty.tag,
            transform: empty.transform,
            children,
        })
    }

    fn create_point_light(
        &mut self,
        root: usize,
        path: Vec<usize>,
        point_light: PointLightBuilder,
    ) -> Entity {
        self.lighting.lights.push(LightUniform {
            position: point_light.transform.position.into(),
            _padding: 0.0,
            color: point_light.color,
            _padding2: 0.0,
            strength: point_light.strength,
            _padding3: [0.0, 0.0, 0.0],
        });
        let light_handle = LightHandle(self.lighting.lights.len() - 1);

        let instance_handle = point_light.mesh_handle.map(|mesh_handle| {
            let mesh_data = self.meshes.get_mut(mesh_handle.0).unwrap();
            mesh_data.light_instances.push(Instance {
                transform: point_light.transform,
            });
            let instance_index = mesh_data.light_instances.len() - 1;
            InstanceHandle {
                mesh: mesh_handle,
                instance_index,
                instance_type: InstanceType::Light,
            }
        });

        let children = self.create_children(root, &path, point_light.children);

        Entity::PointLight(PointLight {
            tag: point_light.tag,
            transform: point_light.transform,
            children,
            color: point_light.color,
            strength: point_light.strength,
            light_handle,
            instance_handle,
        })
    }
}
