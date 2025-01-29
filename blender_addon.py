# -*- coding: utf-8 -*-

bl_info = {
    "name": "Hunyuan3D-2 Generator",
    "author": "Tencent Hunyuan3D",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Hunyuan3D-2 3D Generator",
    "description": "Generate/Texture 3D models from text descriptions or images",
    "category": "3D View",
}

import base64
import os
import tempfile
import threading

import bpy
import requests
from bpy.props import StringProperty, BoolProperty, IntProperty, FloatProperty

class Hunyuan3DProperties(bpy.types.PropertyGroup):
    """Stores addon properties in Blender's scene data"""
    
    # Text input for generation prompt
    prompt: StringProperty(
        name="Text Prompt",
        description="Describe what you want to generate",
        default=""
    )
    
    # API endpoint configuration
    api_url: StringProperty(
        name="API URL",
        description="URL of the Text-to-3D API service",
        default="http://localhost:8080"
    )
    
    # Processing state indicator
    is_processing: BoolProperty(
        name="Processing",
        default=False
    )
    
    # Job tracking ID
    job_id: StringProperty(
        name="Job ID", 
        default=""
    )
    
    # User feedback messages
    status_message: StringProperty(
        name="Status Message",
        default=""
    )
    
    # Image path for image-based generation
    image_path: StringProperty(
        name="Image",
        description="Select an image to upload",
        subtype='FILE_PATH'  # Blender file selector integration
    )
    
    # Octree resolution for 3D generation
    octree_resolution: IntProperty(
        name="Octree Resolution",
        description="Octree resolution for the 3D generation (128-512)",
        default=256,
        min=128,
        max=512,
    )
    
    # Number of inference steps
    num_inference_steps: IntProperty(
        name="Inference Steps",
        description="Number of steps for generation process (20-50)",
        default=20,
        min=20,
        max=50
    )
    
    # Guidance scale parameter
    guidance_scale: FloatProperty(
        name="Guidance Scale",
        description="Controls generation creativity vs accuracy (1.0-10.0)",
        default=5.5,
        min=1.0,
        max=10.0
    )
    
    # Texture generation toggle
    texture: BoolProperty(
        name="Generate Texture",
        description="Enable texture generation for the 3D model",
        default=False
    )

class Hunyuan3DOperator(bpy.types.Operator):
    """Main operator handling generation workflow"""
    
    bl_idname = "object.generate_3d"
    bl_label = "Generate 3D Model"
    bl_description = "Generate 3D model from text, image, or selected mesh"
    
    # Internal state tracking
    job_id = ''
    prompt = ""
    api_url = ""
    image_path = ""
    octree_resolution = 256
    num_inference_steps = 20
    guidance_scale = 5.5
    texture = False
    selected_mesh_base64 = ""
    selected_mesh = None  # Reference to Blender mesh object
    
    # Thread management
    thread = None
    task_finished = False

    def modal(self, context, event):
        """Handles real-time updates and user interruptions"""
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            return {'CANCELLED'}
        
        if self.task_finished:
            print("Generation completed")
            props = context.scene.gen_3d_props
            props.is_processing = False
            self.task_finished = False
            
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        """Starts the generation process"""
        props = context.scene.gen_3d_props
        
        # Store current settings
        self.prompt = props.prompt
        self.api_url = props.api_url
        self.image_path = props.image_path
        self.octree_resolution = props.octree_resolution
        self.num_inference_steps = props.num_inference_steps
        self.guidance_scale = props.guidance_scale
        self.texture = props.texture

        # Validate inputs
        if not self.prompt and not self.image_path:
            self.report({'WARNING'}, "Please enter text or select an image")
            return {'FINISHED'}
        
        # Store selected mesh reference
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                self.selected_mesh = obj
                break
        
        # Export selected mesh if exists
        if self.selected_mesh:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as temp_glb:
                temp_glb.close()
                bpy.ops.export_scene.gltf(filepath=temp_glb.name, use_selection=True)
                
                with open(temp_glb.name, "rb") as f:
                    self.selected_mesh_base64 = base64.b64encode(f.read()).decode()
                
                os.unlink(temp_glb.name)
        
        props.is_processing = True
        
        # Convert relative paths to absolute
        blend_dir = os.path.dirname(bpy.data.filepath)
        if self.image_path.startswith('//'):
            self.image_path = os.path.join(blend_dir, self.image_path[2:])
        
        # Set status messages
        if self.selected_mesh and self.texture:
            props.status_message = "Texturing Selected Mesh...\nThis may take several minutes"
        else:
            mesh_type = 'Textured Mesh' if self.texture else 'White Mesh'
            prompt_type = 'Text Prompt' if self.prompt else 'Image'
            props.status_message = f"Generating {mesh_type} with {prompt_type}...\nPlease wait"
        
        # Start generation thread
        self.thread = threading.Thread(target=self.generate_model)
        self.thread.start()
        
        # Set up modal operator
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def generate_model(self):
        """Main generation logic running in background thread"""
        print("Starting generation...")
        base_url = self.api_url.rstrip('/')
        
        try:
            if self.selected_mesh_base64 and self.texture:
                # Texturing existing mesh
                if self.image_path and os.path.exists(self.image_path):
                    with open(self.image_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode()
                    
                    response = requests.post(
                        f"{base_url}/generate",
                        json={
                            "mesh": self.selected_mesh_base64,
                            "image": img_b64,
                            "octree_resolution": self.octree_resolution,
                            "num_inference_steps": self.num_inference_steps,
                            "guidance_scale": self.guidance_scale,
                            "texture": self.texture
                        },
                    )
                else:
                    response = requests.post(
                        f"{base_url}/generate",
                        json={
                            "mesh": self.selected_mesh_base64,
                            "text": self.prompt,
                            "octree_resolution": self.octree_resolution,
                            "num_inference_steps": self.num_inference_steps,
                            "guidance_scale": self.guidance_scale,
                            "texture": self.texture
                        },
                    )
            else:
                # New generation from image/text
                if self.image_path:
                    if not os.path.exists(self.image_path):
                        self.report({'ERROR'}, f"Image not found: {self.image_path}")
                        raise FileNotFoundError(self.image_path)
                    
                    with open(self.image_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode()
                    
                    response = requests.post(
                        f"{base_url}/generate",
                        json={
                            "image": img_b64,
                            "octree_resolution": self.octree_resolution,
                            "num_inference_steps": self.num_inference_steps,
                            "guidance_scale": self.guidance_scale,
                            "texture": self.texture
                        },
                    )
                else:
                    response = requests.post(
                        f"{base_url}/generate",
                        json={
                            "text": self.prompt,
                            "octree_resolution": self.octree_resolution,
                            "num_inference_steps": self.num_inference_steps,
                            "guidance_scale": self.guidance_scale,
                            "texture": self.texture
                        },
                    )
            
            # Handle API response
            if response.status_code != 200:
                self.report({'ERROR'}, f"API Error: {response.text}")
                return
            
            # Save and import generated model
            with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as f:
                f.write(response.content)
                f.close()
                
                def import_handler():
                    bpy.ops.import_scene.gltf(filepath=f.name)
                    os.unlink(f.name)
                    
                    new_obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
                    if new_obj and self.selected_mesh and self.texture:
                        # Match transforms to original mesh
                        new_obj.location = self.selected_mesh.location
                        new_obj.rotation_euler = self.selected_mesh.rotation_euler
                        new_obj.scale = self.selected_mesh.scale
                        
                        # Hide original mesh
                        self.selected_mesh.hide_set(True)
                        self.selected_mesh.hide_render = True
                
                bpy.app.timers.register(import_handler)
                
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
        finally:
            self.task_finished = True
            self.selected_mesh_base64 = ""

class Hunyuan3DPanel(bpy.types.Panel):
    """UI panel in 3D View sidebar"""
    
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Hunyuan3D-2'
    bl_label = 'Hunyuan3D-2 Generator'

    def draw(self, context):
        """Renders panel UI elements"""
        layout = self.layout
        props = context.scene.gen_3d_props
        
        # API Configuration
        layout.prop(props, "api_url")
        
        # Input Section
        layout.label(text="Inputs:")
        layout.prop(props, "prompt")
        layout.prop(props, "image_path")
        
        # Generation Parameters
        layout.label(text="Parameters:")
        layout.prop(props, "octree_resolution")
        layout.prop(props, "num_inference_steps")
        layout.prop(props, "guidance_scale")
        layout.prop(props, "texture")
        
        # Generate Button
        row = layout.row()
        row.enabled = not props.is_processing
        row.operator("object.generate_3d")
        
        # Status Display
        if props.is_processing:
            if props.status_message:
                for line in props.status_message.split("\n"):
                    layout.label(text=line)
            else:
                layout.label(text="Processing...")

# Registration
classes = (Hunyuan3DProperties, Hunyuan3DOperator, Hunyuan3DPanel)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.gen_3d_props = bpy.props.PointerProperty(type=Hunyuan3DProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.gen_3d_props

if __name__ == "__main__":
    register()
