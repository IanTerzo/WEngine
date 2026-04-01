use nalgebra::{Isometry3, Perspective3, Translation, UnitQuaternion, Vector3};
use rapier3d::prelude::{ColliderBuilder, ColliderHandle};
use std::collections::HashMap;

use crate::{
    camera::CameraUniform,
    entity::{
        Camera, DynamicBody, Empty, Entity, EntityHandle, KinematicBody, MeshInstance, PointLight,
        StaticBody,
        builder::{
            CameraBuilder, DynamicBodyBuilder, EmptyBuilder, EntityBuilder, KinematicBodyBuilder,
            MeshInstanceBuilder, PointLightBuilder, StaticBodyBuilder,
        },
    },
    instance::{Instance, InstanceHandle, InstanceType},
    lightning::{LightHandle, LightUniform},
    mesh::MeshData,
    physics::{ColliderConfig, PhysicsWorld},
    renderer::OPENGL_TO_WGPU_MATRIX,
};

pub fn spawn(
    entities: &mut Vec<Entity>,
    meshes: &mut Vec<MeshData>,
    collider_enitity_pairs: &mut HashMap<ColliderHandle, EntityHandle>,
    physics_world: &mut PhysicsWorld,
    queue: &wgpu::Queue,
    camera_uniform: &mut CameraUniform,
    camera_buffer: &wgpu::Buffer,
    config: &wgpu::SurfaceConfiguration,
    lights: &mut Vec<LightUniform>,
    light_buffer: &wgpu::Buffer,
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
        lights,
        light_buffer,
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
    lights: &mut Vec<LightUniform>,
    light_buffer: &wgpu::Buffer,
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
            lights,
            light_buffer,
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
            lights,
            light_buffer,
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
            lights,
            light_buffer,
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
            lights,
            light_buffer,
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
            lights,
            light_buffer,
            empty,
        ),
        EntityBuilder::PointLight(point_light) => create_point_light(
            meshes,
            collider_enitity_pairs,
            physics_world,
            queue,
            camera_uniform,
            camera_buffer,
            config,
            entity_root_index,
            path,
            lights,
            light_buffer,
            point_light,
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
    lights: &mut Vec<LightUniform>,
    light_buffer: &wgpu::Buffer,
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
        mesh_data.standard_instances.push(Instance {
            transform: body.transform,
        });

        let instance_index = mesh_data.standard_instances.len() - 1;

        instance_handle = Some(InstanceHandle {
            mesh: mesh_handle,
            instance_index,
            instance_type: InstanceType::Standard,
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
                lights,
                light_buffer,
                child,
            )
        })
        .collect();

    Entity::DynamicBody(DynamicBody {
        tag: body.tag,
        transform: body.transform,
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
    lights: &mut Vec<LightUniform>,
    light_buffer: &wgpu::Buffer,
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
        mesh_data.standard_instances.push(Instance {
            transform: body.transform,
        });

        let instance_index = mesh_data.standard_instances.len() - 1;

        instance_handle = Some(InstanceHandle {
            mesh: mesh_handle,
            instance_index,
            instance_type: InstanceType::Standard,
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
                lights,
                light_buffer,
                child,
            )
        })
        .collect();

    Entity::StaticBody(StaticBody {
        tag: body.tag,
        transform: body.transform,
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
    lights: &mut Vec<LightUniform>,
    light_buffer: &wgpu::Buffer,
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
        mesh_data.standard_instances.push(Instance {
            transform: body.transform,
        });

        let instance_index = mesh_data.standard_instances.len() - 1;

        instance_handle = Some(InstanceHandle {
            mesh: mesh_handle,
            instance_index,
            instance_type: InstanceType::Standard,
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
                lights,
                light_buffer,
                child,
            )
        })
        .collect();

    Entity::KinematicBody(KinematicBody {
        tag: body.tag,
        transform: body.transform,
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
    lights: &mut Vec<LightUniform>,
    light_buffer: &wgpu::Buffer,
    mesh_instance: MeshInstanceBuilder,
) -> Entity {
    let mesh_data = meshes.get_mut(mesh_instance.mesh_handle.0).unwrap();
    mesh_data.standard_instances.push(Instance {
        transform: mesh_instance.transform,
    });

    let instance_index = mesh_data.standard_instances.len() - 1;

    let instance_handle = InstanceHandle {
        mesh: mesh_instance.mesh_handle,
        instance_index,
        instance_type: InstanceType::Standard,
    };

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
                lights,
                light_buffer,
                child,
            )
        })
        .collect();

    Entity::MeshInstance(MeshInstance {
        tag: mesh_instance.tag,
        instance_handle: instance_handle,
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
    camera_uniform.view_position = camera.transform.position.to_homogeneous().into();

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
    lights: &mut Vec<LightUniform>,
    light_buffer: &wgpu::Buffer,
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
                lights,
                light_buffer,
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

fn create_point_light(
    meshes: &mut Vec<MeshData>,
    collider_enitity_pairs: &mut HashMap<ColliderHandle, EntityHandle>,
    physics_world: &mut PhysicsWorld,
    queue: &wgpu::Queue,
    camera_uniform: &mut CameraUniform,
    camera_buffer: &wgpu::Buffer,
    config: &wgpu::SurfaceConfiguration,
    root: usize,
    path: Vec<usize>,
    lights: &mut Vec<LightUniform>,
    light_buffer: &wgpu::Buffer,
    point_light: PointLightBuilder,
) -> Entity {
    let light = LightUniform {
        position: point_light.transform.position.into(),
        _padding: 0.0,
        color: point_light.color,
        _padding2: 0.0,
        strength: point_light.strength,
        _padding3: [0.0, 0.0, 0.0],
    };

    lights.push(light);

    let light_handle = LightHandle(lights.len() - 1);

    queue.write_buffer(&light_buffer, 0, bytemuck::cast_slice(&lights));

    let mut instance_handle: Option<InstanceHandle> = None;

    if let Some(mesh_handle) = point_light.mesh_handle {
        let mesh_data = meshes.get_mut(mesh_handle.0).unwrap();
        mesh_data.light_instances.push(Instance {
            transform: point_light.transform,
        });

        let instance_index = mesh_data.light_instances.len() - 1;

        instance_handle = Some(InstanceHandle {
            mesh: mesh_handle,
            instance_index,
            instance_type: InstanceType::Light,
        })
    }

    let child_infos: Vec<_> = point_light
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
                lights,
                light_buffer,
                child,
            )
        })
        .collect();

    Entity::PointLight(PointLight {
        tag: point_light.tag,
        transform: point_light.transform,
        children: child_infos,
        color: point_light.color,
        strenght: point_light.strength,
        light_handle,
        instance_handle,
    })
}
