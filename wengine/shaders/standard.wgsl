struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) normal_matrix_0: vec3<f32>,
    @location(10) normal_matrix_1: vec3<f32>,
    @location(11) normal_matrix_2: vec3<f32>,
};

// Vertex shader

struct Camera {
 	view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> camera_uniform: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    let normal_matrix = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.world_normal = normal_matrix * model.normal;
    var world_position: vec4<f32> = model_matrix * vec4<f32>(model.position, 1.0);
    out.world_position = world_position.xyz;
    out.clip_position = camera_uniform.view_proj * world_position;
    return out;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

struct CameraUniform {
 	view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}
@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct Light {
    position: vec3<f32>,
    _pad1: f32,
    color: vec3<f32>,
    _pad2: f32,
    strength: f32,
}
@group(2) @binding(0)
var<storage, read> lights: array<Light>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    let object_color =
        textureSample(t_diffuse, s_diffuse, in.tex_coords);

    let base_radius = 40.0;

    let N = normalize(in.world_normal);
    let V = normalize(camera.view_pos.xyz - in.world_position);

    let k_a: f32 = 0.05;
    let k_d: f32 = 1.0;
    let k_s: f32 = 0.5;
    let shininess: f32 = 32.0;

    var ambient: vec3<f32>  = vec3<f32>(0.0);
    var diffuse: vec3<f32>  = vec3<f32>(0.0);
    var specular: vec3<f32> = vec3<f32>(0.0);

    for (var i = 0u; i < arrayLength(&lights); i++) {

        let light = lights[i];

        let to_light = light.position - in.world_position;
        let distance = length(to_light);
        let L = normalize(to_light);

        let ambient_radius  = base_radius * 1.5 * light.strength;
        let diffuse_radius  = base_radius * 1.0 * light.strength;
        let specular_radius = base_radius * 0.6 * light.strength;

        let ambient_intensity  = smoothstep(ambient_radius,  0.0, distance);
        let diffuse_intensity  = smoothstep(diffuse_radius,  0.0, distance);
        let specular_intensity = smoothstep(specular_radius, 0.0, distance);

        ambient += light.color * ambient_intensity * k_a;

        let diff = max(dot(N, L), 0.0);
        diffuse += light.color * diff * diffuse_intensity * k_d;

        let H = normalize(L + V);
        let spec = pow(max(dot(N, H), 0.0), shininess);
        specular += light.color * spec * specular_intensity * k_s;
    }

    let final_color =
        object_color.xyz * (ambient + diffuse) + specular;

    return vec4<f32>(final_color, object_color.a);
}
