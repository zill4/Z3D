from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
from ursina.lights import PointLight, AmbientLight
from pathlib import Path
import os

class ArtGalleryViewer:
    def __init__(self):
        # Initialize engine with gallery settings
        self.app = Ursina(
            title='3D Art Gallery',
            borderless=False,
            fullscreen=False,
            size=(1600, 900)
        )
        
        # Configure window
        window.exit_button.visible = False
        window.fps_counter.enabled = True
        
        # Setup gallery environment
        self.setup_gallery_space()
        self.setup_lighting()
        self.load_artwork()
        self.setup_controls()
        
    def setup_gallery_space(self):
        """Create a clean gallery environment"""
        # Floor
        Entity(
            model='plane',
            scale=(20, 1, 20),
            texture='white_cube',
            texture_scale=(4, 4),
            color=color.rgb(200, 200, 200),
            collider='box'
        )
        
        # Ceiling
        Entity(
            model='plane',
            scale=(20, 1, 20),
            position=(0, 4, 0),
            rotation=(180, 0, 0),
            color=color.rgb(180, 180, 180)
        )
        
        # Walls
        self.walls = [
            Entity(model='cube', scale=(20, 5, 0.1), position=(0, 2.5, 10), color=color.rgb(220, 220, 220)),
            Entity(model='cube', scale=(20, 5, 0.1), position=(0, 2.5, -10), color=color.rgb(220, 220, 220)),
            Entity(model='cube', scale=(0.1, 5, 20), position=(10, 2.5, 0), color=color.rgb(220, 220, 220)),
            Entity(model='cube', scale=(0.1, 5, 20), position=(-10, 2.5, 0), color=color.rgb(220, 220, 220))
        ]
        
    def setup_lighting(self):
        """Create gallery-quality lighting"""
        # Ambient lighting
        AmbientLight(color=color.rgb(200, 200, 200))
        
        # Main spotlight
        DirectionalLight(
            y=2.5, z=2,
            shadows=True,
            rotation=(45, 45, 45),
            color=color.rgb(255, 255, 255)
        )
        
        # Fill lights
        SpotLight(
            position=(-3, 3, -3),
            color=color.rgb(200, 200, 200),
            shadows=True
        )
        SpotLight(
            position=(3, 3, -3),
            color=color.rgb(200, 200, 200),
            shadows=True
        )
        
    def get_generation_path(self):
        """Get the path to generation_4"""
        return Path('output/generation_4')

    def load_artwork(self):
        """Load and display the 3D model with debug visuals"""
        gen_dir = Path('output/generation_4')  # Use forward slashes
        
        if not gen_dir.exists():
            print(f"Generation directory {gen_dir} not found")
            return

        # Create a visible marker at target position
        marker = Entity(
            model='cube',
            color=color.red,
            scale=(0.5, 0.5, 0.5),
            position=(0, 1.5, 3)
        )
        
        try:
            # Check if files exist before loading
            obj_path = gen_dir / 'output.obj'
            texture_path = gen_dir / 'texture.png'
            
            if not obj_path.exists():
                print(f"OBJ file not found at: {obj_path}")
                raise FileNotFoundError(f"Missing OBJ file: {obj_path}")
            
            if not texture_path.exists():
                print(f"Texture file not found at: {texture_path}")
                raise FileNotFoundError(f"Missing texture file: {texture_path}")
            
            print(f"Loading model from: {obj_path}")
            print(f"Loading texture from: {texture_path}")
            
            # Try loading with explicit texture
            self.artwork = Entity(
                model=str(obj_path),
                texture=str(texture_path),
                scale=5.0,
                position=(0, 1.5, 3),
                rotation=(0, 180, 0),
                collider='box',
                double_sided=True
            )
            print(f"Loaded model at {self.artwork.position} with scale {self.artwork.scale}")
            
            # Add wireframe overlay
            self.artwork_wireframe = Entity(
                model=self.artwork.model,
                position=self.artwork.position,
                rotation=self.artwork.rotation,
                scale=self.artwork.scale,
                color=color.cyan,
                wireframe=True
            )
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            # Fallback with visible cube
            self.artwork = Entity(
                model='cube',
                position=(0, 1.5, 3),
                scale=(2, 2, 2),
                color=color.magenta,
                collider='box'
            )
            print("Created fallback cube")
            
            # Add debug sphere to mark intended position
            Entity(
                model='sphere',
                scale=(0.5, 0.5, 0.5),
                position=(0, 1.5, 3),
                color=color.yellow
            )

    def setup_controls(self):
        """Configure controls with debug features"""
        self.player = FirstPersonController(
            position=(0, 1.8, 0),
            rotation=(0, 180, 0),
            gravity=0.5,
            speed=3,
            mouse_sensitivity=Vec2(40, 40)
        )
        
        # Debug camera settings
        camera.clip_plane_near = 0.01
        camera.clip_plane_far = 100
        camera.fov = 90
        
        # Debug text
        self.debug_text = Text(
            text="Press 'O' to toggle wireframe\nPress 'P' for position info",
            position=(-0.8, 0.4),
            origin=(-0.5, 0.5),
            scale=0.8,
            color=color.red
        )
        
        def input(key):
            if key == 'o':
                self.artwork_wireframe.enabled = not self.artwork_wireframe.enabled
            if key == 'p':
                print(f"Player position: {self.player.position}")
                print(f"Model position: {self.artwork.position}")
            
        self.input = input
        
    def run(self):
        """Start the gallery viewer"""
        self.app.run()

if __name__ == '__main__':
    gallery = ArtGalleryViewer()
    gallery.run()