$rawFiles = Get-ChildItem "C:\Local3DGenProject\assets\raw3D\*.ply"

foreach ($file in $rawFiles) {
    $outputName = $file.BaseName + "_cleaned.obj"
    $outputPath = Join-Path "C:\Local3DGenProject\assets\cleaned3D" $outputName
    
    & "C:\Local3DGenProject\tools\Meshlab\meshlab.exe" -i $file.FullName -o $outputPath -s .\scripts\poisson_rec.mlx
} 