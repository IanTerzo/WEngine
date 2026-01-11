// Entity builders

use anyhow::anyhow;
use nalgebra::{Isometry3, Matrix4, Perspective3, Translation, UnitQuaternion, Vector3};
use rapier3d::prelude::{ColliderBuilder, ColliderHandle};
use std::collections::HashMap;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

use crate::{
    CameraUniform, Instance, InstanceHandle, Transform,
    model::{MeshData, MeshHandle},
    physics::{ColliderConfig, PhysicsWorld},
};

pub enum EntityBuilder {
    DynamicBody(DynamicBodyBuilder),
    StaticBody(StaticBodyBuilder),
    KinematicBody(KinematicBodyBuilder),
    MeshInstance(MeshInstanceBuilder),
    Camera(CameraBuilder),
    Empty(EmptyBuilder),
}

impl EntityBuilder {
    pub fn dynamic_body(transform: Transform) -> DynamicBodyBuilder {
        DynamicBodyBuilder::new(transform)
    }

    pub fn static_body(transform: Transform) -> StaticBodyBuilder {
        StaticBodyBuilder::new(transform)
    }

    pub fn kinematic_body(transform: Transform) -> KinematicBodyBuilder {
        KinematicBodyBuilder::new(transform)
    }

    pub fn mesh_instance(mesh_handle: MeshHandle, transform: Transform) -> MeshInstanceBuilder {
        MeshInstanceBuilder::new(mesh_handle, transform)
    }

    pub fn camera(transform: Transform) -> CameraBuilder {
        CameraBuilder::new(transform)
    }

    pub fn empty(transform: Transform) -> EmptyBuilder {
        EmptyBuilder::new(transform)
    }
}

pub struct DynamicBodyBuilder {
    pub tag: Option<String>,
    pub mesh_handle: Option<MeshHandle>,
    pub transform: Transform,
    pub children: Vec<EntityBuilder>,
    pub collider: Option<ColliderConfig>,
    pub linear_velocity: Vector3<f32>,
    pub angular_velocity: Vector3<f32>,
    pub mass: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub gravity_scale: f32,
    pub can_sleep: bool,
}

pub struct StaticBodyBuilder {
    pub tag: Option<String>,
    pub mesh_handle: Option<MeshHandle>,
    pub transform: Transform,
    pub children: Vec<EntityBuilder>,
    pub collider: Option<ColliderConfig>,
}

pub struct KinematicBodyBuilder {
    pub tag: Option<String>,
    pub mesh_handle: Option<MeshHandle>,
    pub transform: Transform,
    pub children: Vec<EntityBuilder>,
    pub collider: Option<ColliderConfig>,
    pub linear_velocity: Vector3<f32>,
    pub angular_velocity: Vector3<f32>,
}

pub struct MeshInstanceBuilder {
    pub tag: Option<String>,
    pub mesh_handle: MeshHandle,
    pub transform: Transform,
    pub children: Vec<EntityBuilder>,
}

pub struct CameraBuilder {
    pub tag: Option<String>,
    pub transform: Transform,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

pub struct EmptyBuilder {
    pub tag: Option<String>,
    pub transform: Transform,
    pub children: Vec<EntityBuilder>,
}

impl DynamicBodyBuilder {
    pub fn new(transform: Transform) -> Self {
        Self {
            tag: None,
            mesh_handle: None,
            transform,
            children: vec![],
            collider: None,
            linear_velocity: Vector3::zeros(),
            angular_velocity: Vector3::zeros(),
            mass: 1.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            gravity_scale: 1.0,
            can_sleep: true,
        }
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    pub fn mesh(mut self, mesh: MeshHandle) -> Self {
        self.mesh_handle = Some(mesh);
        self
    }

    pub fn add_child(mut self, child: impl Into<EntityBuilder>) -> Self {
        self.children.push(child.into());
        self
    }

    pub fn collider_ball(mut self, radius: f32) -> Self {
        self.collider = Some(ColliderConfig::Ball { radius });
        self
    }

    pub fn collider_capsule(mut self, half_height: f32, radius: f32) -> Self {
        self.collider = Some(ColliderConfig::Capsule {
            half_height,
            radius,
        });
        self
    }

    pub fn collider_cuboid(mut self, half_extents: Vector3<f32>) -> Self {
        self.collider = Some(ColliderConfig::Cuboid { half_extents });
        self
    }

    pub fn collider_cylinder(mut self, half_height: f32, radius: f32) -> Self {
        self.collider = Some(ColliderConfig::Cylinder {
            half_height,
            radius,
        });
        self
    }

    pub fn linear_velocity(mut self, velocity: Vector3<f32>) -> Self {
        self.linear_velocity = velocity;
        self
    }

    pub fn angular_velocity(mut self, velocity: Vector3<f32>) -> Self {
        self.angular_velocity = velocity;
        self
    }

    pub fn mass(mut self, mass: f32) -> Self {
        self.mass = mass;
        self
    }

    pub fn linear_damping(mut self, damping: f32) -> Self {
        self.linear_damping = damping;
        self
    }

    pub fn angular_damping(mut self, damping: f32) -> Self {
        self.angular_damping = damping;
        self
    }

    pub fn gravity_scale(mut self, scale: f32) -> Self {
        self.gravity_scale = scale;
        self
    }

    pub fn can_sleep(mut self, can_sleep: bool) -> Self {
        self.can_sleep = can_sleep;
        self
    }
}

impl StaticBodyBuilder {
    pub fn new(transform: Transform) -> Self {
        Self {
            tag: None,
            mesh_handle: None,
            transform,
            children: vec![],
            collider: None,
        }
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    pub fn mesh(mut self, mesh: MeshHandle) -> Self {
        self.mesh_handle = Some(mesh);
        self
    }

    pub fn add_child(mut self, child: impl Into<EntityBuilder>) -> Self {
        self.children.push(child.into());
        self
    }

    pub fn collider_ball(mut self, radius: f32) -> Self {
        self.collider = Some(ColliderConfig::Ball { radius });
        self
    }

    pub fn collider_capsule(mut self, half_height: f32, radius: f32) -> Self {
        self.collider = Some(ColliderConfig::Capsule {
            half_height,
            radius,
        });
        self
    }

    pub fn collider_cuboid(mut self, half_extents: Vector3<f32>) -> Self {
        self.collider = Some(ColliderConfig::Cuboid { half_extents });
        self
    }

    pub fn collider_cylinder(mut self, half_height: f32, radius: f32) -> Self {
        self.collider = Some(ColliderConfig::Cylinder {
            half_height,
            radius,
        });
        self
    }
}

impl KinematicBodyBuilder {
    pub fn new(transform: Transform) -> Self {
        Self {
            tag: None,
            mesh_handle: None,
            transform,
            children: vec![],
            collider: None,
            linear_velocity: Vector3::zeros(),
            angular_velocity: Vector3::zeros(),
        }
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    pub fn mesh(mut self, mesh: MeshHandle) -> Self {
        self.mesh_handle = Some(mesh);
        self
    }

    pub fn add_child(mut self, child: impl Into<EntityBuilder>) -> Self {
        self.children.push(child.into());
        self
    }

    pub fn collider_ball(mut self, radius: f32) -> Self {
        self.collider = Some(ColliderConfig::Ball { radius });
        self
    }

    pub fn collider_capsule(mut self, half_height: f32, radius: f32) -> Self {
        self.collider = Some(ColliderConfig::Capsule {
            half_height,
            radius,
        });
        self
    }

    pub fn collider_cuboid(mut self, half_extents: Vector3<f32>) -> Self {
        self.collider = Some(ColliderConfig::Cuboid { half_extents });
        self
    }

    pub fn collider_cylinder(mut self, half_height: f32, radius: f32) -> Self {
        self.collider = Some(ColliderConfig::Cylinder {
            half_height,
            radius,
        });
        self
    }

    pub fn linear_velocity(mut self, velocity: Vector3<f32>) -> Self {
        self.linear_velocity = velocity;
        self
    }

    pub fn angular_velocity(mut self, velocity: Vector3<f32>) -> Self {
        self.angular_velocity = velocity;
        self
    }
}

impl MeshInstanceBuilder {
    pub fn new(mesh_handle: MeshHandle, transform: Transform) -> Self {
        Self {
            tag: None,
            mesh_handle,
            transform,
            children: vec![],
        }
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    pub fn add_child(mut self, child: impl Into<EntityBuilder>) -> Self {
        self.children.push(child.into());
        self
    }
}

impl CameraBuilder {
    pub fn new(transform: Transform) -> Self {
        Self {
            tag: None,
            transform,
            fov: 45.0,
            near: 0.1,
            far: 100.0,
        }
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    pub fn fov(mut self, fov: f32) -> Self {
        self.fov = fov;
        self
    }

    pub fn near(mut self, near: f32) -> Self {
        self.near = near;
        self
    }

    pub fn far(mut self, far: f32) -> Self {
        self.far = far;
        self
    }
}

impl EmptyBuilder {
    pub fn new(transform: Transform) -> Self {
        Self {
            tag: None,
            transform,
            children: vec![],
        }
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    pub fn add_child(mut self, child: impl Into<EntityBuilder>) -> Self {
        self.children.push(child.into());
        self
    }
}

impl From<DynamicBodyBuilder> for EntityBuilder {
    fn from(body: DynamicBodyBuilder) -> Self {
        EntityBuilder::DynamicBody(body)
    }
}

impl From<StaticBodyBuilder> for EntityBuilder {
    fn from(body: StaticBodyBuilder) -> Self {
        EntityBuilder::StaticBody(body)
    }
}

impl From<KinematicBodyBuilder> for EntityBuilder {
    fn from(body: KinematicBodyBuilder) -> Self {
        EntityBuilder::KinematicBody(body)
    }
}

impl From<CameraBuilder> for EntityBuilder {
    fn from(camera: CameraBuilder) -> Self {
        EntityBuilder::Camera(camera)
    }
}

impl From<MeshInstanceBuilder> for EntityBuilder {
    fn from(mesh_instance: MeshInstanceBuilder) -> Self {
        EntityBuilder::MeshInstance(mesh_instance)
    }
}
impl From<EmptyBuilder> for EntityBuilder {
    fn from(empty: EmptyBuilder) -> Self {
        EntityBuilder::Empty(empty)
    }
}

// Engine entity type
// After spawning the entity using Entity, we store it's "EntityInfo", which contains the actual information that is used for rendering.
// Data contains information only relevant to the engine, and entity_state contains info that is also relevant to the user and is the user facing part.

#[derive(Clone, Debug)]
pub struct DynamicBody {
    pub tag: Option<String>,
    pub instance_handle: Option<InstanceHandle>,
    pub rigid_body_handle: rapier3d::prelude::RigidBodyHandle,
    pub children: Vec<Entity>,
}

#[derive(Clone, Debug)]
pub struct StaticBody {
    pub tag: Option<String>,
    pub instance_handle: Option<InstanceHandle>,
    pub rigid_body_handle: rapier3d::prelude::RigidBodyHandle,
    pub children: Vec<Entity>,
}

#[derive(Clone, Debug)]
pub struct KinematicBody {
    pub tag: Option<String>,
    pub instance_handle: Option<InstanceHandle>,
    pub rigid_body_handle: rapier3d::prelude::RigidBodyHandle,
    pub children: Vec<Entity>,
}

#[derive(Clone, Debug)]
pub struct Camera {
    pub tag: Option<String>,
    pub transform: Transform,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

#[derive(Clone, Debug)]
pub struct MeshInstance {
    pub tag: Option<String>,
    pub instance_handle: InstanceHandle,
    pub transform: Transform,
    pub children: Vec<Entity>,
}

#[derive(Clone, Debug)]
pub struct Empty {
    pub tag: Option<String>,
    pub transform: Transform,
    pub children: Vec<Entity>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EntityHandle {
    root: usize,
    path: Vec<usize>,
}

#[derive(Clone, Debug)]
pub enum Entity {
    DynamicBody(DynamicBody),
    StaticBody(StaticBody),
    KinematicBody(KinematicBody),
    Camera(Camera),
    MeshInstance(MeshInstance),
    Empty(Empty),
}

fn children_mut(entity: &mut Entity) -> Option<&mut Vec<Entity>> {
    match entity {
        Entity::DynamicBody(e) => Some(&mut e.children),
        Entity::StaticBody(e) => Some(&mut e.children),
        Entity::KinematicBody(e) => Some(&mut e.children),
        Entity::MeshInstance(e) => Some(&mut e.children),
        Entity::Empty(e) => Some(&mut e.children),
        Entity::Camera(_) => None, // cameras have no children
    }
}

pub fn get_entity_from_handle<'a>(
    entities: &'a mut Vec<Entity>,
    entity_handle: EntityHandle,
) -> anyhow::Result<&'a mut Entity> {
    let mut entity: &mut Entity = entities
        .get_mut(entity_handle.root)
        .ok_or_else(|| anyhow!("Invalid root index: {}", entity_handle.root))?;

    for &index in &entity_handle.path {
        let children =
            children_mut(entity).ok_or_else(|| anyhow!("Entity in path has no children"))?;
        entity = children
            .get_mut(index)
            .ok_or_else(|| anyhow!("Invalid child index: {}", index))?;
    }
    Ok(entity)
}

pub fn spawn(
    entities: &mut Vec<Entity>,
    meshes: &mut Vec<MeshData>,
    collider_enitity_pairs: &mut HashMap<ColliderHandle, EntityHandle>,
    physics_world: &mut PhysicsWorld,
    queue: &wgpu::Queue,
    camera_uniform: &mut CameraUniform,
    camera_buffer: &wgpu::Buffer,
    config: &wgpu::SurfaceConfiguration,
    entity: impl Into<EntityBuilder>,
) -> EntityHandle {
    let entity = create(
        meshes,
        collider_enitity_pairs,
        physics_world,
        queue,
        camera_uniform,
        camera_buffer,
        config,
        entities.len(),
        vec![],
        entity,
    );
    entities.push(entity);

    EntityHandle {
        root: entities.len() - 1,
        path: vec![],
    }
}

fn create(
    meshes: &mut Vec<MeshData>,
    collider_enitity_pairs: &mut HashMap<ColliderHandle, EntityHandle>,
    physics_world: &mut PhysicsWorld,
    queue: &wgpu::Queue,
    camera_uniform: &mut CameraUniform,
    camera_buffer: &wgpu::Buffer,
    config: &wgpu::SurfaceConfiguration,
    entity_root_index: usize,
    path: Vec<usize>,
    entity: impl Into<EntityBuilder>,
) -> Entity {
    let entity = entity.into();

    match entity {
        EntityBuilder::DynamicBody(dynamic) => create_dynamic_rigidbody(
            meshes,
            collider_enitity_pairs,
            physics_world,
            queue,
            camera_uniform,
            camera_buffer,
            config,
            entity_root_index,
            path,
            dynamic,
        ),
        EntityBuilder::StaticBody(static_body) => create_static_rigidbody(
            meshes,
            collider_enitity_pairs,
            physics_world,
            queue,
            camera_uniform,
            camera_buffer,
            config,
            entity_root_index,
            path,
            static_body,
        ),
        EntityBuilder::KinematicBody(kinematic) => create_kinematic_rigidbody(
            meshes,
            collider_enitity_pairs,
            physics_world,
            queue,
            camera_uniform,
            camera_buffer,
            config,
            entity_root_index,
            path,
            kinematic,
        ),
        EntityBuilder::MeshInstance(mesh_instance) => create_mesh_instance(
            meshes,
            collider_enitity_pairs,
            physics_world,
            queue,
            camera_uniform,
            camera_buffer,
            config,
            entity_root_index,
            path,
            mesh_instance,
        ),
        EntityBuilder::Camera(camera) => {
            create_camera(queue, camera_uniform, camera_buffer, config, camera)
        }
        EntityBuilder::Empty(empty) => create_empty(
            meshes,
            collider_enitity_pairs,
            physics_world,
            queue,
            camera_uniform,
            camera_buffer,
            config,
            entity_root_index,
            path,
            empty,
        ),
    }
}

fn create_dynamic_rigidbody(
    meshes: &mut Vec<MeshData>,
    collider_enitity_pairs: &mut HashMap<ColliderHandle, EntityHandle>,
    physics_world: &mut PhysicsWorld,
    queue: &wgpu::Queue,
    camera_uniform: &mut CameraUniform,
    camera_buffer: &wgpu::Buffer,
    config: &wgpu::SurfaceConfiguration,
    root: usize,
    path: Vec<usize>,
    body: DynamicBodyBuilder,
) -> Entity {
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

    let mut instance_handle: Option<InstanceHandle> = None;

    if let Some(mesh_handle) = body.mesh_handle {
        let mesh_data = meshes.get_mut(mesh_handle.0).unwrap();
        mesh_data.instances.push(Instance {
            transform: body.transform,
        });

        let instance_index = mesh_data.instances.len() - 1;

        instance_handle = Some(InstanceHandle {
            mesh: mesh_handle,
            instance_index,
        })
    }

    let rigid_body_handle = physics_world.rigid_body_set.insert(rigid_body);

    if let Some(collider_config) = body.collider {
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
        let collider_handle = physics_world.collider_set.insert_with_parent(
            collider,
            rigid_body_handle,
            &mut physics_world.rigid_body_set,
        );

        let entity_handle = EntityHandle {
            root,
            path: path.clone(),
        };

        collider_enitity_pairs.insert(collider_handle, entity_handle);
    }

    let child_infos: Vec<_> = body
        .children
        .into_iter()
        .enumerate()
        .map(|(i, child)| {
            let mut path = path.clone();
            path.push(i);
            create(
                meshes,
                collider_enitity_pairs,
                physics_world,
                queue,
                camera_uniform,
                camera_buffer,
                config,
                root,
                path,
                child,
            )
        })
        .collect();

    Entity::DynamicBody(DynamicBody {
        tag: body.tag,
        instance_handle,
        rigid_body_handle,
        children: child_infos,
    })
}

fn create_static_rigidbody(
    meshes: &mut Vec<MeshData>,
    collider_enitity_pairs: &mut HashMap<ColliderHandle, EntityHandle>,
    physics_world: &mut PhysicsWorld,
    queue: &wgpu::Queue,
    camera_uniform: &mut CameraUniform,
    camera_buffer: &wgpu::Buffer,
    config: &wgpu::SurfaceConfiguration,
    root: usize,
    path: Vec<usize>,
    body: StaticBodyBuilder,
) -> Entity {
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

    let mut instance_handle: Option<InstanceHandle> = None;

    if let Some(mesh_handle) = body.mesh_handle {
        let mesh_data = meshes.get_mut(mesh_handle.0).unwrap();
        mesh_data.instances.push(Instance {
            transform: body.transform,
        });

        let instance_index = mesh_data.instances.len() - 1;

        instance_handle = Some(InstanceHandle {
            mesh: mesh_handle,
            instance_index,
        })
    }

    let rigid_body_handle = physics_world.rigid_body_set.insert(rigid_body);

    if let Some(collider_config) = body.collider {
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
        let collider_handle = physics_world.collider_set.insert_with_parent(
            collider,
            rigid_body_handle,
            &mut physics_world.rigid_body_set,
        );

        let entity_handle = EntityHandle {
            root,
            path: path.clone(),
        };

        collider_enitity_pairs.insert(collider_handle, entity_handle);
    }

    let child_infos: Vec<_> = body
        .children
        .into_iter()
        .enumerate()
        .map(|(i, child)| {
            let mut path = path.clone();
            path.push(i);
            create(
                meshes,
                collider_enitity_pairs,
                physics_world,
                queue,
                camera_uniform,
                camera_buffer,
                config,
                root,
                path,
                child,
            )
        })
        .collect();

    Entity::StaticBody(StaticBody {
        tag: body.tag,
        instance_handle,
        rigid_body_handle,
        children: child_infos,
    })
}

fn create_kinematic_rigidbody(
    meshes: &mut Vec<MeshData>,
    collider_enitity_pairs: &mut HashMap<ColliderHandle, EntityHandle>,
    physics_world: &mut PhysicsWorld,
    queue: &wgpu::Queue,
    camera_uniform: &mut CameraUniform,
    camera_buffer: &wgpu::Buffer,
    config: &wgpu::SurfaceConfiguration,
    root: usize,
    path: Vec<usize>,
    body: KinematicBodyBuilder,
) -> Entity {
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

    let mut instance_handle: Option<InstanceHandle> = None;

    if let Some(mesh_handle) = body.mesh_handle {
        let mesh_data = meshes.get_mut(mesh_handle.0).unwrap();
        mesh_data.instances.push(Instance {
            transform: body.transform,
        });

        let instance_index = mesh_data.instances.len() - 1;

        instance_handle = Some(InstanceHandle {
            mesh: mesh_handle,
            instance_index,
        })
    }

    let rigid_body_handle = physics_world.rigid_body_set.insert(rigid_body);

    if let Some(collider_config) = body.collider {
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
        let collider_handle = physics_world.collider_set.insert_with_parent(
            collider,
            rigid_body_handle,
            &mut physics_world.rigid_body_set,
        );

        let entity_handle = EntityHandle {
            root,
            path: path.clone(),
        };

        collider_enitity_pairs.insert(collider_handle, entity_handle);
    }

    let child_infos: Vec<_> = body
        .children
        .into_iter()
        .enumerate()
        .map(|(i, child)| {
            let mut path = path.clone();
            path.push(i);
            create(
                meshes,
                collider_enitity_pairs,
                physics_world,
                queue,
                camera_uniform,
                camera_buffer,
                config,
                root,
                path,
                child,
            )
        })
        .collect();

    Entity::KinematicBody(KinematicBody {
        tag: body.tag,
        instance_handle,
        rigid_body_handle,
        children: child_infos,
    })
}

fn create_mesh_instance(
    meshes: &mut Vec<MeshData>,
    collider_enitity_pairs: &mut HashMap<ColliderHandle, EntityHandle>,
    physics_world: &mut PhysicsWorld,
    queue: &wgpu::Queue,
    camera_uniform: &mut CameraUniform,
    camera_buffer: &wgpu::Buffer,
    config: &wgpu::SurfaceConfiguration,
    root: usize,
    path: Vec<usize>,
    mesh_instance: MeshInstanceBuilder,
) -> Entity {
    let mesh_data = meshes.get_mut(mesh_instance.mesh_handle.0).unwrap();
    mesh_data.instances.push(Instance {
        transform: mesh_instance.transform,
    });

    let instance_index = mesh_data.instances.len() - 1;

    let child_infos: Vec<_> = mesh_instance
        .children
        .into_iter()
        .enumerate()
        .map(|(i, child)| {
            let mut path = path.clone();
            path.push(i);

            create(
                meshes,
                collider_enitity_pairs,
                physics_world,
                queue,
                camera_uniform,
                camera_buffer,
                config,
                root,
                path,
                child,
            )
        })
        .collect();

    Entity::MeshInstance(MeshInstance {
        tag: mesh_instance.tag,
        instance_handle: InstanceHandle {
            mesh: mesh_instance.mesh_handle,
            instance_index,
        },
        transform: mesh_instance.transform,
        children: child_infos,
    })
}

fn create_camera(
    queue: &wgpu::Queue,
    camera_uniform: &mut CameraUniform,
    camera_buffer: &wgpu::Buffer,
    config: &wgpu::SurfaceConfiguration,
    camera: CameraBuilder,
) -> Entity {
    // Start by placing the camera at the spawned position

    let iso = Isometry3::from_parts(
        Translation::from(camera.transform.position),
        UnitQuaternion::from_quaternion(camera.transform.rotation),
    );

    let view = iso.inverse().to_homogeneous();

    let aspect = config.width as f32 / config.height as f32;

    let proj = Perspective3::new(aspect, camera.fov.to_radians(), camera.near, camera.far)
        .to_homogeneous();

    camera_uniform.view_proj = (OPENGL_TO_WGPU_MATRIX * proj * view).into();

    queue.write_buffer(&camera_buffer, 0, bytemuck::cast_slice(&[*camera_uniform]));

    // Then we create the camera entity

    Entity::Camera(Camera {
        tag: camera.tag,
        transform: camera.transform,
        fov: camera.fov,
        near: camera.near,
        far: camera.far,
    })
}

fn create_empty(
    meshes: &mut Vec<MeshData>,
    collider_enitity_pairs: &mut HashMap<ColliderHandle, EntityHandle>,
    physics_world: &mut PhysicsWorld,
    queue: &wgpu::Queue,
    camera_uniform: &mut CameraUniform,
    camera_buffer: &wgpu::Buffer,
    config: &wgpu::SurfaceConfiguration,
    root: usize,
    path: Vec<usize>,
    empty: EmptyBuilder,
) -> Entity {
    let child_infos: Vec<_> = empty
        .children
        .into_iter()
        .enumerate()
        .map(|(i, child)| {
            let mut path = path.clone();
            path.push(i);

            create(
                meshes,
                collider_enitity_pairs,
                physics_world,
                queue,
                camera_uniform,
                camera_buffer,
                config,
                root,
                path,
                child,
            )
        })
        .collect();

    Entity::Empty(Empty {
        tag: empty.tag,
        transform: empty.transform,
        children: child_infos,
    })
}
