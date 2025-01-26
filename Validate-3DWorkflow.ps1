<#
.SYNOPSIS
Validates text-to-3D workflow environment setup and basic functionality
#>

$SCRIPT_VERSION = "1.0.0"
$exitCode = 0

# [1] Environment Validation
Write-Host "`n[VALIDATION] Checking System Requirements..." -ForegroundColor Cyan

try {
    # Check Python with full path
    $pythonPath = (Get-Command python -ErrorAction Stop).Path
    $pythonVersion = & $pythonPath --version 2>&1
    if (-not ($pythonVersion -match "Python 3\.\d+")) {
        throw "Python 3.x not found"
    }

    # Check CUDA with common paths
    $cudaPaths = @(
        "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe",
        "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe"
    )
    
    $nvcc = $cudaPaths | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $nvcc) {
        throw "CUDA Toolkit not found in standard paths"
    }
    
    Write-Host "[PASS] Core dependencies verified" -ForegroundColor Green
}
catch {
    Write-Host "[FAIL] $_" -ForegroundColor Red
    $exitCode = 1
}

# [2] Repository Structure Check
Write-Host "`n[VALIDATION] Checking Repository Structure..." -ForegroundColor Cyan

$requiredPaths = @(
    "text-to-3D-workflow/steps/step0_setup.md"
    "scripts/text_to_mesh.py"
    "outputs/"
)

try {
    $missing = $requiredPaths | Where-Object { -not (Test-Path $_) }
    if ($missing) {
        throw "Missing paths: $($missing -join ', ')"
    }
    Write-Host "[PASS] Repository structure valid" -ForegroundColor Green
}
catch {
    Write-Host "[FAIL] $_" -ForegroundColor Red
    $exitCode = 1
}

# [3] Basic Functionality Test
Write-Host "`n[VALIDATION] Testing Text-to-Mesh Generation..." -ForegroundColor Cyan

try {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw "Python not in PATH"
    }

    $testOutput = "validation_output/"
    Remove-Item $testOutput -Recurse -ErrorAction SilentlyContinue
    
    python scripts/text_to_mesh.py --prompt "validation cube" --output $testOutput
    
    if (-not (Test-Path "$testOutput/*.obj")) {
        throw "No OBJ files generated"
    }
    
    Write-Host "[PASS] Basic generation successful" -ForegroundColor Green
}
catch {
    Write-Host "[FAIL] $_" -ForegroundColor Red
    $exitCode = 1
}

# Final result
Write-Host "`nValidation completed with exit code $exitCode (0=success)" -ForegroundColor $(if ($exitCode -eq 0) { "Green" } else { "Red" })
exit $exitCode 