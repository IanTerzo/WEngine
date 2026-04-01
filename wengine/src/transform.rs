use nalgebra::{Matrix4, Translation3, UnitQuaternion, Vector3};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transform {
    pub position: Vector3<f32>,
    pub rotation: UnitQuaternion<f32>,
    pub scale: Vector3<f32>,
}

impl Transform {
    pub fn zero() -> Self {
        Transform {
            position: Vector3::new(0.0, 0.0, 0.0),
            rotation: UnitQuaternion::identity(),
            scale: Vector3::new(1.0, 1.0, 1.0),
        }
    }

    pub fn transform(&self, other: &Transform) -> Transform {
        let rotated_offset = self.rotation.transform_vector(&other.position);

        let new_position = self.position + rotated_offset;

        let new_rotation = self.rotation * other.rotation;

        let new_scale = Vector3::new(
            self.scale.x * other.scale.x,
            self.scale.y * other.scale.y,
            self.scale.z * other.scale.z,
        );

        Transform {
            position: new_position,
            rotation: new_rotation,
            scale: new_scale,
        }
    }

    pub fn to_matrix(&self) -> Matrix4<f32> {
        let translation = Translation3::from(self.position).to_homogeneous();
        // make sure the quaternion is treated as a rotation
        let rotation = self.rotation.to_homogeneous();
        let scale = Matrix4::new_nonuniform_scaling(&self.scale);

        translation * rotation * scale
    }
}
