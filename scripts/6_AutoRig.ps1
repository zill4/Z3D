param(
    [string]$InputObj = "C:\Local3DGenProject\assets\cleaned3D\output_cleaned.obj",
    [string]$OutputFbx = "C:\Local3DGenProject\assets\rigged\drone_rigged.fbx"
)

# Generate temporary script
$blenderScript = @"
import bpy

# Clear existing
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import OBJ
bpy.ops.import_scene.obj(filepath='$InputObj')

# Add metarig
bpy.ops.object.armature_human_metarig_add()
metarig = bpy.context.object

# Generate full rig
bpy.ops.object.mode_set(mode='POSE')
bpy.ops.pose.rigify_generate()

# Export FBX
bpy.ops.export_scene.fbx(
    filepath='$OutputFbx',
    use_selection=True,
    add_leaf_bones=False,
    bake_anim_use_all_actions=False
)
"@ | Out-File "temp_rig.py"

# Execute in Blender
& "C:\Program Files\Blender Foundation\Blender 3.6\blender.exe" -b -P temp_rig.py
Remove-Item temp_rig.py 