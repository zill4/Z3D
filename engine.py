from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
from ursina.lights import PointLight, AmbientLight
from pathlib import Path
import os
from panda3d.core import Filename, Texture, NodePath, TransparencyAttrib, TextureStage, SamplerState, TexGenAttrib
from direct.actor.Actor import Actor
import traceback
from PIL import Image
import numpy as np

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
        """Get the path to generation_27"""
        return Path('output/generation_27')

    def load_artwork(self):
        """Load and display the 3D model with debug visuals"""
        gen_dir = Path('output/generation_27')
        
        try:
            # Define paths
            model_paths = {
                'obj': gen_dir / 'base_mesh.obj',
                'glb': gen_dir / 'untextured_mesh.glb',
                'processed': gen_dir / 'processed_base.glb',
            }
            texture_path = gen_dir / 'texture.png'
            
            print("Attempting to load models from multiple sources...")
            
            # Load and process texture using PIL first
            if texture_path.exists():
                print(f"\nLoading texture from: {texture_path}")
                # Load and convert texture to RGBA
                pil_image = Image.open(texture_path)
                pil_image = pil_image.convert('RGBA')
                
                # Ensure dimensions are power of 2
                target_size = (512, 512)
                pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
                
                # Save processed texture
                processed_texture_path = gen_dir / 'processed_texture.png'
                pil_image.save(processed_texture_path)
                print(f"Processed texture saved to: {processed_texture_path}")
                
                # Load processed texture
                tex = loader.loadTexture(Filename.from_os_specific(str(processed_texture_path)))
                if tex:
                    tex.setFormat(Texture.FRgba)
                    tex.setMagfilter(SamplerState.FT_nearest)
                    tex.setMinfilter(SamplerState.FT_nearest)
                    print("Texture loaded successfully")

            # Try different texture application methods
            
            # Method 1: Direct texture assignment (left)
            print(f"\nTrying OBJ with direct texture: {model_paths['obj']}")
            self.obj_model = Entity(
                model=str(model_paths['obj']),
                texture=str(processed_texture_path),  # Direct texture assignment
                position=(-6, 1.5, 3),
                rotation=(0, 180, 0),
                scale=1
            )
            Entity(model='sphere', color=color.blue, scale=0.3, position=(-6, 1.5, 3))
            print("Direct texture model created at (-6, 1.5, 3)")

            # Method 2: Manual UV mapping (center-left)
            print("\nTrying manual UV mapping")
            self.uv_model = Entity(
                model=str(model_paths['obj']),
                position=(-3, 1.5, 3),
                rotation=(0, 180, 0),
                scale=1
            )
            if tex:
                ts = TextureStage('ts')
                ts.setMode(TextureStage.MModulate)
                ts.setSort(0)
                self.uv_model.model.setTexGen(ts, TexGenAttrib.MWorldPosition)  # Generate UVs based on world position
                self.uv_model.model.setTexture(ts, tex)
            Entity(model='sphere', color=color.red, scale=0.3, position=(-3, 1.5, 3))
            print("UV mapped model created at (-3, 1.5, 3)")

            # Method 3: Texture projection (center)
            print("\nTrying texture projection")
            self.projected_model = Entity(
                model=str(model_paths['obj']),
                position=(0, 1.5, 3),
                rotation=(0, 180, 0),
                scale=1
            )
            if tex:
                ts = TextureStage('ts')
                ts.setMode(TextureStage.MDecal)  # Use decal mode for projection
                self.projected_model.model.setTexGen(ts, TexGenAttrib.MEyeSphereMap)  # Spherical projection
                self.projected_model.model.setTexture(ts, tex)
            Entity(model='sphere', color=color.green, scale=0.3, position=(0, 1.5, 3))
            print("Projected texture model created at (0, 1.5, 3)")

            # Method 4: Colored version (center-right)
            print("\nCreating colored version")
            self.colored_model = Entity(
                model=str(model_paths['obj']),
                position=(3, 1.5, 3),
                rotation=(0, 180, 0),
                scale=1,
                color=color.pink
            )
            Entity(model='sphere', color=color.yellow, scale=0.3, position=(3, 1.5, 3))
            print("Colored model created at (3, 1.5, 3)")

            # Method 5: Wireframe version (right)
            print("\nCreating wireframe version")
            self.wireframe_model = Entity(
                model=str(model_paths['obj']),
                position=(6, 1.5, 3),
                rotation=(0, 180, 0),
                scale=1,
                color=color.cyan,
                wireframe=True
            )
            Entity(model='sphere', color=color.magenta, scale=0.3, position=(6, 1.5, 3))
            print("Wireframe model created at (6, 1.5, 3)")

            # Store reference to main artwork for existing controls
            self.artwork = self.projected_model
            
            print("\nDebug positions:")
            print(f"Direct texture: Vec3(-6, 1.5, 3)")
            print(f"UV mapped: Vec3(-3, 1.5, 3)")
            print(f"Projected: Vec3(0, 1.5, 3)")
            print(f"Colored: Vec3(3, 1.5, 3)")
            print(f"Wireframe: Vec3(6, 1.5, 3)")
            
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            traceback.print_exc()
            print("Creating fallback cube...")
            
            # Fallback cube
            self.artwork = Entity(
                model='cube',
                position=(0, 1.5, 3),
                scale=2,
                color=color.magenta
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
        # elif key == 'o':
        #     if hasattr(self, 'artwork_wireframe'):
        #         self.artwork_wireframe.enabled = not self.artwork_wireframe.enabled
        #         print(f"Wireframe {'enabled' if self.artwork_wireframe.enabled else 'disabled'}")
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