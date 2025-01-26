# Install core components per InstallPlan.txt
$pythonUrl = "https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe"
$blenderUrl = "https://download.blender.org/release/Blender3.6/blender-3.6.2-windows-x64.msi"
$meshlabUrl = "https://github.com/cnr-isti-vclab/meshlab/releases/download/Meshlab-2022.12/MeshLab2022.12-windows.exe"

# Install Python 3.10.9
Write-Host "Downloading Python installer..."
Invoke-WebRequest $pythonUrl -OutFile $env:TEMP\python-installer.exe
if (-not (Test-Path $env:TEMP\python-installer.exe)) {
    throw "Python installer download failed"
}

Start-Process -Wait -FilePath $env:TEMP\python-installer.exe -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1 TargetDir="C:\Python310"'

# Verify installation
if (-not (Test-Path "C:\Python310\python.exe")) {
    throw "Python installation failed"
}

& "C:\Python310\python.exe" -m venv C:\Local3DGenProject\.venv

# Install Blender
Start-Process msiexec.exe -Wait -ArgumentList "/i $blenderUrl /qn"

# Install Meshlab (portable mode)
New-Item -Path "C:\Local3DGenProject\tools\Meshlab" -ItemType Directory
Invoke-WebRequest $meshlabUrl -OutFile $env:TEMP\meshlab.exe
Start-Process -Wait -FilePath $env:TEMP\meshlab.exe -ArgumentList "--portable C:\Local3DGenProject\tools\Meshlab" 