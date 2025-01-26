param(
    [string]$BvhFile = "C:\Local3DGenProject\assets\animations\combat_idle.bvh",
    [string]$InputFbx = "C:\Local3DGenProject\assets\rigged\drone_rigged.fbx",
    [string]$OutputFbx = "C:\Local3DGenProject\assets\animated\drone_animated.fbx"
)

$animationScript = @"
import bpy

# Import rigged model
bpy.ops.import_scene.fbx(filepath='$InputFbx')

# Import animation
bpy.ops.import_anim.bvh(filepath='$BvhFile', 
    rotate_mode='NATIVE', 
    update_scene_fps=True,
    global_scale=1.0)

# Configure NLA
obj = bpy.context.object
if obj.animation_data:
    obj.animation_data.action = bpy.data.actions[-1]

# Export animated model
bpy.ops.export_scene.fbx(
    filepath='$OutputFbx',
    bake_anim_use_nla_strips=True,
    bake_anim_use_all_actions=False,
    add_leaf_bones=False
)
"@ | Out-File "temp_anim.py"

& "C:\Program Files\Blender Foundation\Blender 3.6\blender.exe" -b -P temp_anim.py
Remove-Item temp_anim.py 