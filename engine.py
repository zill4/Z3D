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
            vsync=True,
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
        """Get the path to generation_10"""
        return Path('output/generation_10')

    def load_artwork(self):
        """Load and display the 3D model with debug visuals"""
        gen_dir = Path('output/generation_10')
        
        if not gen_dir.exists():
            print(f"Generation directory {gen_dir} not found")
            return

        # Create visible markers to help with orientation
        Entity(model='sphere', color=color.yellow, scale=0.3, position=(0, 1.5, 3))  # Model position
        Entity(model='sphere', color=color.red, scale=0.3, position=(0, 0, 0))      # Origin
        Entity(model='sphere', color=color.green, scale=0.3, position=(0, 0, 1))    # +Z direction

        try:
            # Load base_mesh.obj and texture.png specifically
            obj_path = gen_dir / 'base_mesh.obj'
            texture_path = gen_dir / 'texture.png'
            
            print(f"Looking for OBJ at: {obj_path}")
            print(f"Looking for texture at: {texture_path}")
            
            if not obj_path.exists():
                raise FileNotFoundError(f"Missing OBJ file: {obj_path}")
            if not texture_path.exists():
                raise FileNotFoundError(f"Missing texture file: {texture_path}")
            
            print("Found both model and texture files, attempting to load...")
            
            # Try loading with explicit paths
            self.artwork = Entity(
                model=str(obj_path),
                texture=str(texture_path),
                scale=1.0,
                position=(0, 1.5, 3),
                rotation=(90, 0, 0),  # Adjust for coordinate system difference
                double_sided=True
            )
            print(f"Model loaded: {self.artwork}")
            
            # Add model inspection
            print(f"Model vertices: {len(self.artwork.model.vertices)}")
            print(f"Model triangles: {len(self.artwork.model.triangles)}")
            print(f"Model UVs: {len(self.artwork.model.uvs) if hasattr(self.artwork.model, 'uvs') else None}")
            
            # Add wireframe overlay
            self.artwork_wireframe = Entity(
                model=self.artwork.model,
                position=self.artwork.position,
                rotation=self.artwork.rotation,
                scale=self.artwork.scale,
                color=color.cyan,
                wireframe=True
            )
            print("Wireframe overlay created")
            
        except Exception as e:
            print(f"Model loading failed with error: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            
            # Fallback cube
            self.artwork = Entity(
                model='cube',
                position=(0, 1.5, 3),
                scale=(2, 2, 2),
                color=color.magenta,
                collider='box'
            )
            print("Created fallback cube")

    def setup_controls(self):
        """Configure controls with debug features"""
        self.player = FirstPersonController(
            position=(0, 1.8, -5),
            rotation=(0, 0, 0),
            gravity=0.5,
            speed=3,
            mouse_sensitivity=Vec2(40, 40)
        )
        
        # Adjust camera settings
        camera.clip_plane_near = 0.1
        camera.clip_plane_far = 1000
        camera.fov = 90
        
        # Add debug crosshair and controls text
        Text('+', color=color.red, origin=(0, 0), scale=2)
        
        # Add controls text in the top-left corner
        self.controls_text = Text(
            text='Controls:\nWASD - Move\nMouse - Look\nO - Toggle Wireframe\nP - Debug Info\nESC - Quit',
            position=(-0.85, 0.45),
            scale=1.5,
            color=color.white
        ) 
        
        # Bind input handler to key events
        self.key_handler = Entity()
        self.key_handler.input = self.handle_input

    def handle_input(self, key):
        """Custom input handler"""
        if key == 'escape':
            print("Quitting application")
            application.quit()
        elif key == 'o':
            if hasattr(self, 'artwork_wireframe'):
                self.artwork_wireframe.enabled = not self.artwork_wireframe.enabled
                print(f"Wireframe {'enabled' if self.artwork_wireframe.enabled else 'disabled'}")
        elif key == 'p':
            print(f"\nDebug Info:")
            print(f"Player position: {self.player.position}")
            print(f"Player rotation: {self.player.rotation}")
            if hasattr(self, 'artwork'):
                print(f"Model position: {self.artwork.position}")
                print(f"Model visible: {self.artwork.visible}")
                print(f"Model scale: {self.artwork.scale}")

    def run(self):
        """Start the gallery viewer"""
        self.app.run()

if __name__ == '__main__':
    gallery = ArtGalleryViewer()
    gallery.run()