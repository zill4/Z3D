# Clone repositories per ModelsPlan.txt
$modelsDir = "C:\Local3DGenProject\models"

# Stable Point-E
git clone https://github.com/openai/point-e $modelsDir/point-e

# GET3D checkpoints
$get3dWeights = @(
    "https://api.ngc.nvidia.com/v2/models/nvidia/research/get3d/versions/1/zip",
    "https://api.ngc.nvidia.com/v2/models/nvidia/research/get3d-pretrained/versions/1/zip"
)

foreach ($url in $get3dWeights) {
    $output = "$modelsDir\get3d-$(Get-Date -Format 'yyyyMMdd').zip"
    Write-Host "Downloading $url..."
    Invoke-WebRequest $url -OutFile $output
    if (-not (Test-Path $output)) {
        throw "Failed to download model weights from $url"
    }

    Expand-Archive -Path $output -DestinationPath "$modelsDir\get3d"
    if (-not (Test-Path "$modelsDir\get3d\*.pth")) {
        throw "Model checkpoint extraction failed"
    }
} 