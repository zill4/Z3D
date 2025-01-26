$animatedModels = Get-ChildItem "C:\Local3DGenProject\assets\animated\*.fbx"
$unityDir = "C:\Local3DGenProject\game_engine\UnityProject\Assets"

foreach ($model in $animatedModels) {
    Copy-Item $model.FullName -Destination $unityDir
    Write-Host "Exported $($model.Name) to Unity project"
} 