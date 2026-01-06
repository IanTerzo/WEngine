// Entity builders

use nalgebra::{Isometry3, Matrix4, Perspective3, Translation, UnitQuaternion, Vector3};
use rapier3d::prelude::ColliderBuilder;

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

pub enum Entity {
    DynamicBody(DynamicBody),
    StaticBody(StaticBody),
    KinematicBody(KinematicBody),
    MeshInstance(MeshInstance),
    Camera(Camera),
    Empty(Empty),
}

impl Entity {
    pub fn dynamic_body(transform: Transform) -> DynamicBody {
        DynamicBody::new(transform)
    }

    pub fn static_body(transform: Transform) -> StaticBody {
        StaticBody::new(transform)
    }

    pub fn kinematic_body(transform: Transform) -> KinematicBody {
        KinematicBody::new(transform)
    }

    pub fn mesh_instance(mesh_handle: MeshHandle, transform: Transform) -> MeshInstance {
        MeshInstance::new(mesh_handle, transform)
    }

    pub fn camera(transform: Transform) -> Camera {
        Camera::new(transform)
    }

    pub fn empty(transform: Transform) -> Empty {
        Empty::new(transform)
    }
}

pub struct DynamicBody {
    pub mesh_handle: Option<MeshHandle>,
    pub transform: Transform,
    pub children: Vec<Entity>,
    pub collider: Option<ColliderConfig>,
    pub linear_velocity: Vector3<f32>,
    pub angular_velocity: Vector3<f32>,
    pub mass: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub gravity_scale: f32,
    pub can_sleep: bool,
}

pub struct StaticBody {
    pub mesh_handle: Option<MeshHandle>,
    pub transform: Transform,
    pub children: Vec<Entity>,
    pub collider: Option<ColliderConfig>,
}

pub struct KinematicBody {
    pub mesh_handle: Option<MeshHandle>,
    pub transform: Transform,
    pub children: Vec<Entity>,
    pub collider: Option<ColliderConfig>,
    pub linear_velocity: Vector3<f32>,
    pub angular_velocity: Vector3<f32>,
}

pub struct MeshInstance {
    pub mesh_handle: MeshHandle,
    pub transform: Transform,
    pub children: Vec<Entity>,
}

pub struct Camera {
    pub transform: Transform,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

pub struct Empty {
    pub transform: Transform,
    pub children: Vec<Entity>,
}

impl DynamicBody {
    pub fn new(transform: Transform) -> Self {
        Self {
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

    pub fn mesh(mut self, mesh: MeshHandle) -> Self {
        self.mesh_handle = Some(mesh);
        self
    }

    pub fn add_child(mut self, child: impl Into<Entity>) -> Self {
        self.children.push(child.into());
        self
    }

    pub fn children<I>(mut self, children: Vec<impl Into<Entity>>) -> Self {
        for c in children {
            self.children.push(c.into());
        }
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

impl StaticBody {
    pub fn new(transform: Transform) -> Self {
        Self {
            mesh_handle: None,
            transform,
            children: vec![],
            collider: None,
        }
    }

    pub fn mesh(mut self, mesh: MeshHandle) -> Self {
        self.mesh_handle = Some(mesh);
        self
    }

    pub fn add_child(mut self, child: impl Into<Entity>) -> Self {
        self.children.push(child.into());
        self
    }

    pub fn children<I>(mut self, children: Vec<impl Into<Entity>>) -> Self {
        for c in children {
            self.children.push(c.into());
        }
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

impl KinematicBody {
    pub fn new(transform: Transform) -> Self {
        Self {
            mesh_handle: None,
            transform,
            children: vec![],
            collider: None,
            linear_velocity: Vector3::zeros(),
            angular_velocity: Vector3::zeros(),
        }
    }

    pub fn mesh(mut self, mesh: MeshHandle) -> Self {
        self.mesh_handle = Some(mesh);
        self
    }

    pub fn add_child(mut self, child: impl Into<Entity>) -> Self {
        self.children.push(child.into());
        self
    }

    pub fn children<I>(mut self, children: Vec<impl Into<Entity>>) -> Self {
        for c in children {
            self.children.push(c.into());
        }
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

impl MeshInstance {
    pub fn new(mesh_handle: MeshHandle, transform: Transform) -> Self {
        Self {
            mesh_handle,
            transform,
            children: vec![],
        }
    }

    pub fn add_child(mut self, child: impl Into<Entity>) -> Self {
        self.children.push(child.into());
        self
    }

    pub fn children<I>(mut self, children: Vec<impl Into<Entity>>) -> Self {
        for c in children {
            self.children.push(c.into());
        }
        self
    }
}

impl Camera {
    pub fn new(transform: Transform) -> Self {
        Self {
            transform,
            fov: 45.0,
            near: 0.1,
            far: 100.0,
        }
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

impl Empty {
    pub fn new(transform: Transform) -> Self {
        Self {
            transform,
            children: vec![],
        }
    }

    pub fn add_child(mut self, child: impl Into<Entity>) -> Self {
        self.children.push(child.into());
        self
    }

    pub fn children<I>(mut self, children: Vec<impl Into<Entity>>) -> Self {
        for c in children {
            self.children.push(c.into());
        }
        self
    }
}

impl From<DynamicBody> for Entity {
    fn from(body: DynamicBody) -> Self {
        Entity::DynamicBody(body)
    }
}

impl From<StaticBody> for Entity {
    fn from(body: StaticBody) -> Self {
        Entity::StaticBody(body)
    }
}

impl From<KinematicBody> for Entity {
    fn from(body: KinematicBody) -> Self {
        Entity::KinematicBody(body)
    }
}

impl From<Camera> for Entity {
    fn from(camera: Camera) -> Self {
        Entity::Camera(camera)
    }
}

impl From<MeshInstance> for Entity {
    fn from(mesh_instance: MeshInstance) -> Self {
        Entity::MeshInstance(mesh_instance)
    }
}
impl From<Empty> for Entity {
    fn from(empty: Empty) -> Self {
        Entity::Empty(empty)
    }
}

// Engine entity type
// After spawning the entity using Entity, we store it's "EntityInfo", which contains the actual information that is used for rendering.
// Data contains information only relevant to the engine, and entity_state contains info that is also relevant to the user and is the user facing part.

pub struct RigidBodyRef {
    pub instance_handle: Option<InstanceHandle>,
    pub rigid_body_handle: rapier3d::prelude::RigidBodyHandle,
    pub children: Vec<EntityRef>,
}

pub struct CameraRef {
    pub transform: Transform,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

pub struct MeshInstanceRef {
    pub instance_handle: InstanceHandle,
    pub transform: Transform,
    pub children: Vec<EntityRef>,
}

pub struct EmptyRef {
    pub transform: Transform,
    pub children: Vec<EntityRef>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EntityHandle {
    root: usize,
    path: Vec<usize>,
}

pub enum EntityRef {
    DynamicBody(RigidBodyRef),
    StaticBody(RigidBodyRef),
    KinematicBody(RigidBodyRef),
    Camera(CameraRef),
    MeshInstance(MeshInstanceRef),
    Empty(EmptyRef),
}

fn children_mut(entity: &mut EntityRef) -> Option<&mut Vec<EntityRef>> {
    match entity {
        EntityRef::DynamicBody(e) => Some(&mut e.children),
        EntityRef::StaticBody(e) => Some(&mut e.children),
        EntityRef::KinematicBody(e) => Some(&mut e.children),
        EntityRef::MeshInstance(e) => Some(&mut e.children),
        EntityRef::Empty(e) => Some(&mut e.children),
        EntityRef::Camera(_) => None, // cameras have no children
    }
}

pub fn get_entity_from_handle<'a>(
    entities: &'a mut Vec<EntityRef>,
    entity_handle: EntityHandle,
) -> &'a mut EntityRef {
    let mut entity: &mut EntityRef = &mut entities[entity_handle.root];

    for &index in &entity_handle.path {
        let children = children_mut(entity).expect("entity in path has no children");
        entity = &mut children[index];
    }

    entity
}
pub fn spawn(
    entities: &mut Vec<EntityRef>,
    meshes: &mut Vec<MeshData>,
    physics_world: &mut PhysicsWorld,
    queue: &wgpu::Queue,
    camera_uniform: &mut CameraUniform,
    camera_buffer: &wgpu::Buffer,
    config: &wgpu::SurfaceConfiguration,
    entity: impl Into<Entity>,
) -> EntityHandle {
    let entity = create(
        meshes,
        physics_world,
        queue,
        camera_uniform,
        camera_buffer,
        config,
        entity,
    );
    entities.push(entity);

    let entity_index = entities.len() - 1;

    EntityHandle {
        root: entity_index,
        path: vec![],
    }
}

fn create(
    meshes: &mut Vec<MeshData>,
    physics_world: &mut PhysicsWorld,
    queue: &wgpu::Queue,
    camera_uniform: &mut CameraUniform,
    camera_buffer: &wgpu::Buffer,
    config: &wgpu::SurfaceConfiguration,
    entity: impl Into<Entity>,
) -> EntityRef {
    let entity = entity.into();

    match entity {
        Entity::DynamicBody(dynamic) => create_dynamic_rigidbody(
            meshes,
            physics_world,
            queue,
            camera_uniform,
            camera_buffer,
            config,
            dynamic,
        ),
        Entity::StaticBody(static_body) => create_static_rigidbody(
            meshes,
            physics_world,
            queue,
            camera_uniform,
            camera_buffer,
            config,
            static_body,
        ),
        Entity::KinematicBody(kinematic) => create_kinematic_rigidbody(
            meshes,
            physics_world,
            queue,
            camera_uniform,
            camera_buffer,
            config,
            kinematic,
        ),
        Entity::MeshInstance(mesh_instance) => create_mesh_instance(
            meshes,
            physics_world,
            queue,
            camera_uniform,
            camera_buffer,
            config,
            mesh_instance,
        ),
        Entity::Camera(camera) => {
            create_camera(queue, camera_uniform, camera_buffer, config, camera)
        }
        Entity::Empty(empty) => create_empty(
            meshes,
            physics_world,
            queue,
            camera_uniform,
            camera_buffer,
            config,
            empty,
        ),
    }
}

fn create_dynamic_rigidbody(
    meshes: &mut Vec<MeshData>,
    physics_world: &mut PhysicsWorld,
    queue: &wgpu::Queue,
    camera_uniform: &mut CameraUniform,
    camera_buffer: &wgpu::Buffer,
    config: &wgpu::SurfaceConfiguration,
    body: DynamicBody,
) -> EntityRef {
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

    create_rigidbody(
        body.mesh_handle,
        body.transform,
        body.collider,
        body.children,
        meshes,
        physics_world,
        queue,
        camera_uniform,
        camera_buffer,
        config,
        rigid_body,
    )
}

fn create_static_rigidbody(
    meshes: &mut Vec<MeshData>,
    physics_world: &mut PhysicsWorld,
    queue: &wgpu::Queue,
    camera_uniform: &mut CameraUniform,
    camera_buffer: &wgpu::Buffer,
    config: &wgpu::SurfaceConfiguration,
    body: StaticBody,
) -> EntityRef {
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

    create_rigidbody(
        body.mesh_handle,
        body.transform,
        body.collider,
        body.children,
        meshes,
        physics_world,
        queue,
        camera_uniform,
        camera_buffer,
        config,
        rigid_body,
    )
}

fn create_kinematic_rigidbody(
    meshes: &mut Vec<MeshData>,
    physics_world: &mut PhysicsWorld,
    queue: &wgpu::Queue,
    camera_uniform: &mut CameraUniform,
    camera_buffer: &wgpu::Buffer,
    config: &wgpu::SurfaceConfiguration,
    body: KinematicBody,
) -> EntityRef {
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

    create_rigidbody(
        body.mesh_handle,
        body.transform,
        body.collider,
        body.children,
        meshes,
        physics_world,
        queue,
        camera_uniform,
        camera_buffer,
        config,
        rigid_body,
    )
}

fn create_rigidbody(
    mesh_handle: Option<MeshHandle>,
    transform: Transform,
    collider_config: Option<ColliderConfig>,
    children: Vec<Entity>,
    meshes: &mut Vec<MeshData>,
    physics_world: &mut PhysicsWorld,
    queue: &wgpu::Queue,
    camera_uniform: &mut CameraUniform,
    camera_buffer: &wgpu::Buffer,
    config: &wgpu::SurfaceConfiguration,
    rigid_body: rapier3d::prelude::RigidBody,
) -> EntityRef {
    let mut instance_handle: Option<InstanceHandle> = None;

    if let Some(mesh_handle) = mesh_handle {
        let mesh_data = meshes.get_mut(mesh_handle.0).unwrap();
        mesh_data.instances.push(Instance { transform });

        let instance_index = mesh_data.instances.len() - 1;

        instance_handle = Some(InstanceHandle {
            mesh: mesh_handle,
            instance_index,
        })
    }

    let rigid_body_handle = physics_world.rigid_body_set.insert(rigid_body);

    if let Some(collider_config) = collider_config {
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
        physics_world.collider_set.insert_with_parent(
            collider,
            rigid_body_handle,
            &mut physics_world.rigid_body_set,
        );
    }

    let child_infos: Vec<_> = children
        .into_iter()
        .map(|child| {
            create(
                meshes,
                physics_world,
                queue,
                camera_uniform,
                camera_buffer,
                config,
                child,
            )
        })
        .collect();

    EntityRef::DynamicBody(RigidBodyRef {
        instance_handle,
        rigid_body_handle,
        children: child_infos,
    })
}

fn create_mesh_instance(
    meshes: &mut Vec<MeshData>,
    physics_world: &mut PhysicsWorld,
    queue: &wgpu::Queue,
    camera_uniform: &mut CameraUniform,
    camera_buffer: &wgpu::Buffer,
    config: &wgpu::SurfaceConfiguration,
    mesh_instance: MeshInstance,
) -> EntityRef {
    let mesh_data = meshes.get_mut(mesh_instance.mesh_handle.0).unwrap();
    mesh_data.instances.push(Instance {
        transform: mesh_instance.transform,
    });

    let instance_index = mesh_data.instances.len() - 1;

    let child_infos: Vec<_> = mesh_instance
        .children
        .into_iter()
        .map(|child| {
            create(
                meshes,
                physics_world,
                queue,
                camera_uniform,
                camera_buffer,
                config,
                child,
            )
        })
        .collect();

    EntityRef::MeshInstance(MeshInstanceRef {
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
    camera: Camera,
) -> EntityRef {
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

    EntityRef::Camera(CameraRef {
        transform: camera.transform,
        fov: camera.fov,
        near: camera.near,
        far: camera.far,
    })
}

fn create_empty(
    meshes: &mut Vec<MeshData>,
    physics_world: &mut PhysicsWorld,
    queue: &wgpu::Queue,
    camera_uniform: &mut CameraUniform,
    camera_buffer: &wgpu::Buffer,
    config: &wgpu::SurfaceConfiguration,
    empty: Empty,
) -> EntityRef {
    let child_infos: Vec<_> = empty
        .children
        .into_iter()
        .map(|child| {
            create(
                meshes,
                physics_world,
                queue,
                camera_uniform,
                camera_buffer,
                config,
                child,
            )
        })
        .collect();

    EntityRef::Empty(EmptyRef {
        transform: empty.transform,
        children: child_infos,
    })
}
