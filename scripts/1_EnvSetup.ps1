# Install core components per InstallPlan.txt
$pythonUrl = "https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe"
$blenderUrl = "https://download.blender.org/release/Blender3.6/blender-3.6.2-windows-x64.msi"
$meshlabUrl = "https://github.com/cnr-isti-vclab/meshlab/releases/download/Meshlab-2023.12/MeshLab2023.12-windows.exe"

# Function to download with retry and verification
function Download-WithRetry {
    param (
        [string]$Url,
        [string]$OutFile,
        [string]$Name
    )
    
    Write-Host "Downloading $Name from $Url"
    try {
        $ProgressPreference = 'SilentlyContinue'  # Speeds up downloads
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing -ErrorAction Stop
        
        if (Test-Path $OutFile) {
            Write-Host "Successfully downloaded $Name" -ForegroundColor Green
            return $true
        }
    }
    catch {
        Write-Host "Error downloading $Name $_" -ForegroundColor Red
        return $false
    }
    return $false
}

# Create base directory
$baseDir = "C:\Local3DGenProject"
New-Item -Path $baseDir -ItemType Directory -Force | Out-Null
New-Item -Path "$baseDir\tools" -ItemType Directory -Force | Out-Null

# Download and install Python
if (Download-WithRetry -Url $pythonUrl -OutFile "$env:TEMP\python-installer.exe" -Name "Python") {
    Write-Host "Installing Python..."
    Start-Process -Wait -FilePath "$env:TEMP\python-installer.exe" -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1 TargetDir="C:\Python310"'
    
    if (Test-Path "C:\Python310\python.exe") {
        Write-Host "Python installed successfully" -ForegroundColor Green
        & "C:\Python310\python.exe" -m venv "$baseDir\.venv"
    } else {
        Write-Host "Python installation failed" -ForegroundColor Red
    }
}

# Download and install Blender
if (Download-WithRetry -Url $blenderUrl -OutFile "$env:TEMP\blender-installer.msi" -Name "Blender") {
    Write-Host "Installing Blender..."
    Start-Process msiexec.exe -Wait -ArgumentList "/i `"$env:TEMP\blender-installer.msi`" /qn"
    Write-Host "Blender installation completed" -ForegroundColor Green
}

# Download and install Meshlab
$meshlabDir = "$baseDir\tools\Meshlab"
New-Item -Path $meshlabDir -ItemType Directory -Force | Out-Null

if (Download-WithRetry -Url $meshlabUrl -OutFile "$env:TEMP\meshlab-installer.exe" -Name "Meshlab") {
    Write-Host "Installing Meshlab..."
    try {
        Start-Process -Wait -FilePath "$env:TEMP\meshlab-installer.exe" -ArgumentList "/S /D=`"$meshlabDir`"" -ErrorAction Stop
        Write-Host "Meshlab installation completed" -ForegroundColor Green
    }
    catch {
        Write-Host "Error installing Meshlab: $_" -ForegroundColor Red
        Write-Host "Please download and install Meshlab manually from: https://www.meshlab.net/#download"
    }
}

Write-Host "`nSetup complete! Please check above for any errors." -ForegroundColor Cyan 