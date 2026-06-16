param(
  [string]$ExePath = "build-winml\Release\rnnoise_demo.exe",
  [string]$InputPath = "build-winml\Release\madoka1.bin",
  [int]$SampleRate = 44100,
  [int]$BytesPerSample = 2,
  [int]$LogicalProcessors = [Environment]::ProcessorCount,
  [string]$OutputDir = "build-winml\Release",
  [string]$Prefix = "measure"
)

$ErrorActionPreference = "Stop"

function Resolve-RequiredPath($Path, $Name) {
  if (-not (Test-Path -LiteralPath $Path)) {
    throw "$Name not found: $Path"
  }
  return (Resolve-Path -LiteralPath $Path).Path
}

function Invoke-RnnoiseMeasuredRun($Name, $UseWinmlValue, $Exe, $InputFile, $OutDir, $Prefix) {
  $outFile = Join-Path $OutDir "$Prefix`_$Name.raw"
  $stdoutFile = Join-Path $OutDir "$Prefix`_$Name.stdout.txt"
  $stderrFile = Join-Path $OutDir "$Prefix`_$Name.stderr.txt"

  Remove-Item -LiteralPath $outFile, $stdoutFile, $stderrFile -Force -ErrorAction SilentlyContinue

  $psi = [Diagnostics.ProcessStartInfo]::new()
  $psi.FileName = $Exe
  $psi.WorkingDirectory = (Resolve-Path -LiteralPath ".").Path
  $psi.ArgumentList.Add($InputFile)
  $psi.ArgumentList.Add($outFile)
  $psi.UseShellExecute = $false
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.Environment.Remove("RNNOISE_WINML_REQUIRE_NPU") | Out-Null
  $psi.Environment.Remove("RNNOISE_WINML_EP") | Out-Null

  if ($null -eq $UseWinmlValue) {
    $psi.Environment.Remove("RNNOISE_USE_WINML") | Out-Null
  } else {
    $psi.Environment["RNNOISE_USE_WINML"] = $UseWinmlValue
  }

  $proc = [Diagnostics.Process]::new()
  $proc.StartInfo = $psi
  $wall = [Diagnostics.Stopwatch]::StartNew()
  [void]$proc.Start()
  $stdout = $proc.StandardOutput.ReadToEnd()
  $stderr = $proc.StandardError.ReadToEnd()
  $proc.WaitForExit()
  $wall.Stop()

  [IO.File]::WriteAllText($stdoutFile, $stdout)
  [IO.File]::WriteAllText($stderrFile, $stderr)

  [pscustomobject]@{
    Mode = $Name
    ExitCode = $proc.ExitCode
    CpuSeconds = $proc.TotalProcessorTime.TotalSeconds
    WallSeconds = $wall.Elapsed.TotalSeconds
    OutputBytes = if (Test-Path -LiteralPath $outFile) { (Get-Item -LiteralPath $outFile).Length } else { 0 }
    OutputPath = $outFile
    StdoutPath = $stdoutFile
    StderrPath = $stderrFile
    Stderr = $stderr.Trim()
  }
}

$exe = Resolve-RequiredPath $ExePath "Executable"
$inputFile = Resolve-RequiredPath $InputPath "Input file"
$outDir = Resolve-RequiredPath $OutputDir "Output directory"
$inputBytes = (Get-Item -LiteralPath $inputFile).Length
$audioSeconds = $inputBytes / $BytesPerSample / $SampleRate

$runs = @(
  (Invoke-RnnoiseMeasuredRun "winml_on" $null $exe $inputFile $outDir $Prefix),
  (Invoke-RnnoiseMeasuredRun "winml_off" "0" $exe $inputFile $outDir $Prefix)
)

$results = foreach ($run in $runs) {
  $singleCorePercent = $run.CpuSeconds / $audioSeconds * 100.0
  [pscustomobject]@{
    Mode = $run.Mode
    ExitCode = $run.ExitCode
    InputBytes = $inputBytes
    AudioSeconds = [Math]::Round($audioSeconds, 3)
    AudioMinutes = [Math]::Round($audioSeconds / 60.0, 3)
    CpuSeconds = [Math]::Round($run.CpuSeconds, 3)
    WallSeconds = [Math]::Round($run.WallSeconds, 3)
    BatchCpuPercentSingleCore = [Math]::Round($run.CpuSeconds / $run.WallSeconds * 100.0, 3)
    BatchCpuPercentWholeMachine = [Math]::Round($run.CpuSeconds / $run.WallSeconds / $LogicalProcessors * 100.0, 3)
    RealtimeCpuPercentSingleCore = [Math]::Round($singleCorePercent, 3)
    RealtimeCpuPercentWholeMachine = [Math]::Round($singleCorePercent / $LogicalProcessors, 3)
    OutputBytes = $run.OutputBytes
    OutputPath = $run.OutputPath
    StderrPath = $run.StderrPath
    Stderr = $run.Stderr
  }
}

$csvPath = Join-Path $outDir "$Prefix`_cpu_measurements.csv"
$jsonPath = Join-Path $outDir "$Prefix`_cpu_measurements.json"
$results | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $csvPath
$results | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -LiteralPath $jsonPath

$results | Format-List
Write-Host "CSV: $csvPath"
Write-Host "JSON: $jsonPath"
