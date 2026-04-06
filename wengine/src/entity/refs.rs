use anyhow::anyhow;
use nalgebra::{Quaternion, Unit, Vector3};

use crate::{
    entity::{
        Camera, DynamicBody, Empty, EntityHandle, KinematicBody, MeshInstance, PointLight,
        StaticBody,
    },
    physics::PhysicsWorld,
};

// Entity references that are passed when using get_entity, holds information from the scene (EngineState)

#[derive(Debug)]
pub struct StaticBodyRef<'a> {
    pub entity: &'a mut StaticBody,
}

pub struct DynamicBodyRef<'a> {
    pub entity: &'a mut DynamicBody,
    pub physics_world: &'a mut PhysicsWorld,
}

pub struct KinematicBodyRef<'a> {
    pub entity: &'a mut KinematicBody,
    pub physics_world: &'a mut PhysicsWorld,
}

pub struct CameraRef<'a> {
    pub entity: &'a mut Camera,
}

pub struct MeshInstanceRef<'a> {
    pub entity: &'a mut MeshInstance,
}

pub struct EmptyRef<'a> {
    pub entity: &'a mut Empty,
}

pub struct PointLightRef<'a> {
    pub entity: &'a mut PointLight,
}

pub enum EntityRef<'a> {
    DynamicBody(DynamicBodyRef<'a>),
    StaticBody(StaticBodyRef<'a>),
    KinematicBody(KinematicBodyRef<'a>),
    Camera(CameraRef<'a>),
    MeshInstance(MeshInstanceRef<'a>),
    Empty(EmptyRef<'a>),
    PointLight(PointLightRef<'a>),
}

impl<'a> EntityRef<'a> {
    // Convenient functions
    pub fn into_staticbody(self) -> anyhow::Result<StaticBodyRef<'a>> {
        match self {
            EntityRef::StaticBody(body) => Ok(body),
            _ => Err(anyhow!("EntityRef is not a StaticBody")),
        }
    }

    pub fn into_dynamicbody(self) -> anyhow::Result<DynamicBodyRef<'a>> {
        match self {
            EntityRef::DynamicBody(body) => Ok(body),
            _ => Err(anyhow!("EntityRef is not a DynamicBody")),
        }
    }

    pub fn into_kinematicbody(self) -> anyhow::Result<KinematicBodyRef<'a>> {
        match self {
            EntityRef::KinematicBody(body) => Ok(body),
            _ => Err(anyhow!("EntityRef is not a KinematicBody")),
        }
    }

    pub fn into_camera(self) -> anyhow::Result<CameraRef<'a>> {
        match self {
            EntityRef::Camera(body) => Ok(body),
            _ => Err(anyhow!("EntityRef is not a Camera")),
        }
    }

    pub fn into_mesh_instance(self) -> anyhow::Result<MeshInstanceRef<'a>> {
        match self {
            EntityRef::MeshInstance(body) => Ok(body),
            _ => Err(anyhow!("EntityRef is not a MeshInstance")),
        }
    }

    pub fn into_empty(self) -> anyhow::Result<EmptyRef<'a>> {
        match self {
            EntityRef::Empty(body) => Ok(body),
            _ => Err(anyhow!("EntityRef is not a Empty")),
        }
    }

    pub fn into_pointlight(self) -> anyhow::Result<PointLightRef<'a>> {
        match self {
            EntityRef::PointLight(body) => Ok(body),
            _ => Err(anyhow!("EntityRef is not a PointLight")),
        }
    }
}

impl<'a> DynamicBodyRef<'a> {
    pub fn get_child(&'a mut self, n: usize) -> anyhow::Result<EntityHandle> {
        self.entity
            .children
            .get(n)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Child {} not found", n))
    }

    pub fn add_force(&mut self, vector: Vector3<f32>) -> anyhow::Result<()> {
        if let Some(body) = self
            .physics_world
            .rigid_body_set
            .get_mut(self.entity.rigid_body_handle)
        {
            body.add_force(vector, true);

            Ok(())
        } else {
            Err(anyhow!("Failed to find rigidbody associated with entity"))
        }
    }

    pub fn get_rotation(&self) -> anyhow::Result<Unit<Quaternion<f32>>> {
        if let Some(body) = self
            .physics_world
            .rigid_body_set
            .get(self.entity.rigid_body_handle)
        {
            Ok(*body.rotation())
        } else {
            Err(anyhow!("Failed to find rigidbody associated with entity"))
        }
    }

    pub fn get_position(&self) -> anyhow::Result<Vector3<f32>> {
        if let Some(body) = self
            .physics_world
            .rigid_body_set
            .get(self.entity.rigid_body_handle)
        {
            Ok(*body.translation())
        } else {
            Err(anyhow!("Failed to find rigidbody associated with entity"))
        }
    }

    pub fn set_position(&mut self, vector: Vector3<f32>) -> anyhow::Result<()> {
        if let Some(body) = self
            .physics_world
            .rigid_body_set
            .get_mut(self.entity.rigid_body_handle)
        {
            body.set_translation(vector, true);
            Ok(())
        } else {
            Err(anyhow!("Failed to find rigidbody associated with entity"))
        }
    }

    pub fn set_enabled_rotations(
        &mut self,
        enable_x: bool,
        enable_y: bool,
        enable_z: bool,
    ) -> anyhow::Result<()> {
        if let Some(body) = self
            .physics_world
            .rigid_body_set
            .get_mut(self.entity.rigid_body_handle)
        {
            body.set_enabled_rotations(enable_x, enable_y, enable_z, true);
            Ok(())
        } else {
            Err(anyhow!("Failed to find rigidbody associated with entity"))
        }
    }

    pub fn set_linvel(&mut self, vector: Vector3<f32>) -> anyhow::Result<()> {
        if let Some(body) = self
            .physics_world
            .rigid_body_set
            .get_mut(self.entity.rigid_body_handle)
        {
            body.set_linvel(vector, true);
            Ok(())
        } else {
            Err(anyhow!("Failed to find rigidbody associated with entity"))
        }
    }

    pub fn get_linvel(&self) -> anyhow::Result<Vector3<f32>> {
        if let Some(body) = self
            .physics_world
            .rigid_body_set
            .get(self.entity.rigid_body_handle)
        {
            Ok(*body.linvel())
        } else {
            Err(anyhow!("Failed to find rigidbody associated with entity"))
        }
    }

    pub fn set_angvel(&mut self, vector: Vector3<f32>) -> anyhow::Result<()> {
        if let Some(body) = self
            .physics_world
            .rigid_body_set
            .get_mut(self.entity.rigid_body_handle)
        {
            body.set_angvel(vector, true);
            Ok(())
        } else {
            Err(anyhow!("Failed to find rigidbody associated with entity"))
        }
    }

    pub fn get_angvel(&self) -> anyhow::Result<Vector3<f32>> {
        if let Some(body) = self
            .physics_world
            .rigid_body_set
            .get(self.entity.rigid_body_handle)
        {
            Ok(*body.angvel())
        } else {
            Err(anyhow!("Failed to find rigidbody associated with entity"))
        }
    }
}
