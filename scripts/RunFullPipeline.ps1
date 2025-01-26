<#
.SYNOPSIS
Executes complete text-to-3D pipeline from prompt to game engine export

.DESCRIPTION
Combines all pipeline stages into single command with progress monitoring
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Prompt,
    
    [string]$OutputName = "GeneratedAsset"
)

# Phase 1: Environment Validation
function Test-Environment {
    # Verify base directory structure
    $requiredDirs = @('assets/raw3D', 'assets/cleaned3D', 'assets/rigged', 'assets/animated')
    foreach ($dir in $requiredDirs) {
        $fullPath = Join-Path "C:\Local3DGenProject" $dir
        if (-not (Test-Path $fullPath)) {
            New-Item -Path $fullPath -ItemType Directory -Force
        }
    }

    $checks = @(
        @{Name="Python"; Path="C:\Python310\python.exe"},
        @{Name="Blender"; Path="C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"},
        @{Name="Meshlab"; Path="C:\Local3DGenProject\tools\Meshlab\meshlab.exe"}
    )

    foreach ($check in $checks) {
        if (-not (Test-Path $check.Path)) {
            throw "Critical component missing: $($check.Name)"
        }
    }
}

# Phase 2: Pipeline Execution
function Invoke-FullPipeline {
    [CmdletBinding()]
    param()

    # Inference
    .\scripts\4_Inference.ps1 -Prompt $Prompt
    
    # Mesh Processing
    $rawFile = Get-ChildItem "C:\Local3DGenProject\assets\raw3D\$OutputName*.ply" | 
               Sort-Object LastWriteTime | 
               Select-Object -Last 1
    .\scripts\5_MeshCleanup.ps1

    # Rigging
    $cleanedObj = "C:\Local3DGenProject\assets\cleaned3D\$($rawFile.BaseName)_cleaned.obj"
    .\scripts\6_AutoRig.ps1 -InputObj $cleanedObj -OutputFbx "C:\Local3DGenProject\assets\rigged\$OutputName.fbx"

    # Animation
    .\scripts\7_ApplyAnimation.ps1 -InputFbx "C:\Local3DGenProject\assets\rigged\$OutputName.fbx"

    # Export
    .\scripts\8_UnityExport.ps1
}

# Phase 3: Results Presentation
function Show-Results {
    $report = @"
=== Pipeline Results ===
Prompt: $Prompt
Output Directory: C:\Local3DGenProject\game_engine\UnityProject\Assets

Generated Files:
- Point Cloud: $(Get-ChildItem "C:\Local3DGenProject\assets\raw3D\$OutputName*.ply" | % Length | ForEach-Object {"{0:N1}MB" -f ($_/1MB)})
- Cleaned Mesh: $(Get-ChildItem "C:\Local3DGenProject\assets\cleaned3D\$OutputName*.obj" | % Length | ForEach-Object {"{0:N1}MB" -f ($_/1MB)})
- Animated Model: $(Get-ChildItem "C:\Local3DGenProject\assets\animated\$OutputName*.fbx" | % Length | ForEach-Object {"{0:N1}MB" -f ($_/1MB)})

VRAM Utilization:
$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv)
"@

    $report | Out-File "C:\Local3DGenProject\results.txt"
    Invoke-Item "C:\Local3DGenProject\results.txt"
    Invoke-Item "C:\Local3DGenProject\game_engine\UnityProject"
}

# Main Execution
try {
    Test-Environment
    Invoke-FullPipeline -ErrorAction Stop
    Show-Results
}
catch {
    Write-Error "Pipeline failed: $_"
    exit 1
} 