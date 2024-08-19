# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

param(
    [ValidateSet(
        "coreml",
        "mps",
        "xnnpack"
    )]
    [string[]]$pybind,
    [ValidateSet(
        "quantized",
        "quantized_aot"
    )]
    [string[]]$build_kernels
)

# Before doing anything, cd to the directory containing this script.
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Definition)

# Find the names of the python tools to use.
if (-not $env:PYTHON_EXECUTABLE) {
    if (-not $env:CONDA_DEFAULT_ENV -or $env:CONDA_DEFAULT_ENV -eq "base" -or -not (Get-Command python -ErrorAction SilentlyContinue)) {
        $env:PYTHON_EXECUTABLE = "python3"
    } else {
        $env:PYTHON_EXECUTABLE = "python"
    }
}

if ($env:PYTHON_EXECUTABLE -eq "python") {
    $env:PIP_EXECUTABLE = "pip"
} else {
    $env:PIP_EXECUTABLE = "pip3"
}

# Function to check if the current Python version is compatible.
function Test-PythonCompatibility {
    $versionSpecifier = Select-String "^requires-python" pyproject.toml | Select-Object -First 1 | ForEach-Object { $_.Line -replace '.*?"([^"]*)".*', '$1' }
    
    if (-not $versionSpecifier) {
        Write-Warning "Skipping python version check: version range not found"
        return, $true
    }

    # Install the packaging module if necessary.
    if (-not (& $env:PYTHON_EXECUTABLE -c 'import packaging' 2>$null)) {
        & $env:PIP_EXECUTABLE install packaging > $null
    }

    # Compare the current python version to the range in versionSpecifier.
    try {
        $script = @"
import sys
try:
    import packaging.version
    import packaging.specifiers
    import platform

    python_version = packaging.version.parse(platform.python_version())
    version_range = packaging.specifiers.SpecifierSet('$versionSpecifier')
    if python_version not in version_range:
        print(
            'ERROR: ExecuTorch does not support python version '
            + f'{python_version}: must satisfy \"{versionSpecifier}\"',
            file=sys.stderr,
        )
        sys.exit(1)
except Exception as e:
    print(f'WARNING: Skipping python version check: {e}', file=sys.stderr)
    sys.exit(0)
"@

        & $env:PYTHON_EXECUTABLE -c $script > $null
        return, ($LASTEXITCODE -eq 0)
    } catch {
        Write-Warning "Skipping python version check: $($_.Exception.Message)"
        return, $true
    }
}

# Parse options.
$EXECUTORCH_BUILD_PYBIND = "OFF"
$CMAKE_ARGS = ""

if ($pybind)
{
    $EXECUTORCH_BUILD_PYBIND = "ON"
}

foreach ($arg in $pybind) {
    $upper = $arg.ToUpper()
    $CMAKE_ARGS += "-DEXECUTORCH_BUILD_$($upper)=ON "
}

foreach ($arg in $build_kernels) {
    $upper = $arg.ToUpper()
    $CMAKE_ARGS += "-DEXECUTORCH_BUILD_KERNELS_$($upper)=ON "
}

# Install pip packages used by code in the ExecuTorch repo.
$NIGHTLY_VERSION = "dev20240716"
$TORCH_NIGHTLY_URL = "https://download.pytorch.org/whl/nightly/cpu"

$EXIR_REQUIREMENTS = @(
    "torch==2.5.0.$NIGHTLY_VERSION"
    "torchvision==0.20.0.$NIGHTLY_VERSION"
)

$DEVEL_REQUIREMENTS = @(
    "cmake"
    "pip>=23"
    "pyyaml"
    "setuptools>=63"
    "tomli"
    "wheel"
    "zstd"
)

$EXAMPLES_REQUIREMENTS = @(
    "timm==1.0.7"
    "torchaudio==2.4.0.$NIGHTLY_VERSION"
    "torchsr==1.0.4"
    "transformers==4.42.4"
)

$REQUIREMENTS_TO_INSTALL = $EXIR_REQUIREMENTS + $DEVEL_REQUIREMENTS + $EXAMPLES_REQUIREMENTS

& $env:PIP_EXECUTABLE install --extra-index-url $TORCH_NIGHTLY_URL $REQUIREMENTS_TO_INSTALL

# Install executorch pip package.
$env:EXECUTORCH_BUILD_PYBIND = $EXECUTORCH_BUILD_PYBIND
$env:CMAKE_ARGS = $CMAKE_ARGS
$env:CMAKE_BUILD_ARGS = $CMAKE_BUILD_ARGS

& $env:PIP_EXECUTABLE install . --no-build-isolation -v --extra-index-url $TORCH_NIGHTLY_URL