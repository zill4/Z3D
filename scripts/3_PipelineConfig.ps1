# Configure Meshlab batch processing
$meshlabScript = @"
<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Poisson Surface Reconstruction">
  <Param name="OctDepth" value="12" />
  <Param name="SolverDivide" value="8" />
 </filter>
</FilterScript>
"@ | Out-File "C:\Local3DGenProject\scripts\poisson_rec.mlx"

# Blender automation template
@"
import bpy

bpy.ops.import_scene.obj(filepath='C:/Local3DGenProject/assets/cleaned3D/input.obj')
bpy.ops.object.armature_human_metarig_add()
bpy.ops.object.mode_set(mode='POSE')
bpy.ops.pose.rigify_generate()
"@ | Out-File "C:\Local3DGenProject\scripts\auto_rig.py" 