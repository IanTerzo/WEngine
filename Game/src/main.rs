use crate::{level1::Level1, level2::Level2, level3::Level3};
use wengine::{
    app::Runner,
    scene::{EngineEvent, SceneContext},
};
use winit::keyboard::{KeyCode, PhysicalKey};

mod level1;
mod level2;
mod level3;
mod player;

fn handle_scene_switch(event: &EngineEvent, ctx: &mut SceneContext) {
    if let EngineEvent::Key {
        physical_key: PhysicalKey::Code(code),
        pressed: true,
    } = event
    {
        match code {
            KeyCode::Digit1 => ctx.switch_scene(Level1::new()),
            KeyCode::Digit2 => ctx.switch_scene(Level2::new()),
            KeyCode::Digit3 => ctx.switch_scene(Level3::new()),

            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    Runner::new(Level1::new())
        .window_width(1280)
        .window_height(720)
        .title("First person controller")
        .run()
}
