# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script helps build and run C++ tests with CMakeLists.txt.
# It builds and installs the root ExecuTorch package, and then sub-directories.
#
# If no arg is given, it probes all sub-directories containing
# test/CMakeLists.txt. It builds and runs these tests.
# If an arg is given, like `runtime/core/test/`, it runs that directory only.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)) {
    $env:LLVM_PROFDATA = $env:LLVM_PROFDATA -or "xcrun llvm-profdata"
    $env:LLVM_COV = $env:LLVM_COV -or "xcrun llvm-cov"
} elseif ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::OSX)) {
    $env:LLVM_PROFDATA = $env:LLVM_PROFDATA -or "xcrun llvm-profdata"
    $env:LLVM_COV = $env:LLVM_COV -or "xcrun llvm-cov"
} elseif ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Linux)) {
    $env:LLVM_PROFDATA = $env:LLVM_PROFDATA -or "llvm-profdata"
    $env:LLVM_COV = $env:LLVM_COV -or "llvm-cov"
}

function Build-ExecuTorch {
    $BUILD_VULKAN = "OFF"
    if (Get-Command glslc -ErrorAction SilentlyContinue) {
        $BUILD_VULKAN = "ON"
    }
#        -DEXECUTORCH_BUILD_XNNPACK=ON `
        # -DEXECUTORCH_USE_CPP_CODE_COVERAGE=ON `
    cmake . `
        -DCMAKE_INSTALL_PREFIX=cmake-out `
        -DEXECUTORCH_USE_CPP_CODE_COVERAGE=OFF `
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON `
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON `
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON `
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON `
        -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON `
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON `
        -DEXECUTORCH_BUILD_SDK=ON `
        -DEXECUTORCH_BUILD_VULKAN="$BUILD_VULKAN" `
        -DEXECUTORCH_BUILD_XNNPACK=OFF `
        -Bcmake-out
    # cmake --build cmake-out -j9
    cmake --build cmake-out -j9 --target install
    # throw 2
}

function Build-GTest {
    New-Item -ItemType Directory -Force -Path "third-party/googletest/build" | Out-Null
    Push-Location "third-party/googletest/build"
    cmake .. -DCMAKE_INSTALL_PREFIX=.
    #make -j4
    #cmake -S .. -B . -D CMAKE_BUILD_TYPE=Release
    cmake --build .
    #make install
    cmake --install .
    Pop-Location
}

function Export-TestModel {
    python -m test.models.export_program --modules "ModuleAdd,ModuleAddHalf,ModuleDynamicCatUnallocatedIO,ModuleIndex,ModuleLinear,ModuleMultipleEntry,ModuleSimpleTrain" --outdir "cmake-out" 2> $null
    python3 -m test.models.export_delegated_program --modules "ModuleAddMul" --backend_id "StubBackend" --outdir "cmake-out" | Out-Null

    $global:ET_MODULE_ADD_HALF_PATH = Resolve-Path "cmake-out/ModuleAddHalf.pte"
    $global:ET_MODULE_ADD_PATH = Resolve-Path "cmake-out/ModuleAdd.pte"
    $global:ET_MODULE_DYNAMIC_CAT_UNALLOCATED_IO_PATH = Resolve-Path "cmake-out/ModuleDynamicCatUnallocatedIO.pte"
    $global:ET_MODULE_INDEX_PATH = Resolve-Path "cmake-out/ModuleIndex.pte"
    $global:ET_MODULE_LINEAR_CONSTANT_BUFFER_PATH = Resolve-Path "cmake-out/ModuleLinear-no-constant-segment.pte"
    $global:ET_MODULE_LINEAR_CONSTANT_SEGMENT_PATH = Resolve-Path "cmake-out/ModuleLinear.pte"
    $global:ET_MODULE_MULTI_ENTRY_PATH = Resolve-Path "cmake-out/ModuleMultipleEntry.pte"
    $global:ET_MODULE_ADD_MUL_NOSEGMENTS_DA1024_PATH = Resolve-Path "cmake-out/ModuleAddMul-nosegments-da1024.pte"
    $global:ET_MODULE_ADD_MUL_NOSEGMENTS_PATH = Resolve-Path "cmake-out/ModuleAddMul-nosegments.pte"
    $global:ET_MODULE_ADD_MUL_PATH = Resolve-Path "cmake-out/ModuleAddMul.pte"
    $global:ET_MODULE_SIMPLE_TRAIN_PATH = Resolve-Path "cmake-out/ModuleSimpleTrain.pte"
}

function Build-AndRunTest {
    param ($TestDir)
    write-warning "$(join-path (Get-Location) third-party/googletest/build)"
    # -DEXECUTORCH_USE_CPP_CODE_COVERAGE=ON `
    cmake $TestDir `
        -DCMAKE_BUILD_TYPE=Debug `
        -DCMAKE_INSTALL_PREFIX=cmake-out `
        -DEXECUTORCH_USE_CPP_CODE_COVERAGE=OFF `
        -DCMAKE_PREFIX_PATH="$(join-path (Get-Location) third-party/googletest/build)" `
        -B(join-path cmake-out $TestDir)
    #throw 1
    #cmake --build cmake-out/$TestDir -j9
    cmake --build (join-path cmake-out $TestDir)

    if ($TestDir -match ".*examples/models/llama2/tokenizer.*") {
        $env:RESOURCES_PATH = Resolve-Path "examples/models/llama2/tokenizer/test/resources"
    } elseif ($TestDir -match ".*extension/llm/tokenizer.*") {
        $env:RESOURCES_PATH = Resolve-Path "extension/llm/tokenizer/test/resources"
    } else {
        $env:RESOURCES_PATH = Resolve-Path "extension/module/test/resources"
    }

    $global:TEST_BINARY_LIST = ""
    Get-ChildItem "cmake-out/$TestDir/*test" | ForEach-Object {
        if (Test-Path $_) {
            $env:LLVM_PROFILE_FILE = "cmake-out/$($_.Name).profraw"
            & $_.FullName
            $global:TEST_BINARY_LIST += " -object $($_.FullName)"
        }
    }
}

function Report-Coverage {
    & $env:LLVM_PROFDATA merge -sparse cmake-out/*.profraw -o cmake-out/merged.profdata
    & $env:LLVM_COV report -instr-profile=cmake-out/merged.profdata $global:TEST_BINARY_LIST
}

function Probe-Tests {
    # This function finds the set of directories that contain C++ tests
    # CMakeLists.txt rules, that are buildable using Build-AndRunTest
    $dirs = @(
        "backends",
        "examples",
        "extension",
        "kernels",
        "runtime",
        "schema",
        "devtools",
        "test"
    )
    
    $pwdLength = (join-path $pwd \).length

    $dirs | ForEach-Object { 
        #Get-ChildItem -Recurse -Filter 'CMakeLists.txt' -Include "$($_)\test" -Exclude "$($_)\third-party" |
        #ForEach-Object { $_.DirectoryName } |
        #Sort-Object -Unique
        # ls -Recurse $_ | %{ls -Recurse -Path $_.fullname -Filter test -Directory} | ?{ls -Path $_.fullname -Filter CMakeLists.txt} | Select-Object -Property FullName
        $results = ls -Recurse $_ -Filter test -Directory | ?{(ls -Path $_.fullname -Filter CMakeLists.txt|measure).count -gt 0}
        $results | %{$_.fullname.substring($pwdLength)}
    }
}

#Build-ExecuTorch
# throw 1
#Build-GTest
#Export-TestModel
#Probe-Tests
#throw 1
if (-not $args) {
    Write-Host "Running all directories:"
    $testDirs = Probe-Tests
    foreach ($testDir in $testDirs) {
        Build-AndRunTest $testDir
    }
} else {
    Build-AndRunTest $args[0]
}

try {
    Report-Coverage
} catch {
    Write-Host "Reporting coverage failed, continuing..."
}
