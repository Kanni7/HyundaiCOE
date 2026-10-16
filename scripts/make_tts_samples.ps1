# Generates SYNTHETIC speech samples with the Windows OneCore TTS voices (offline).
# These are a smoke test for the ASR path only; they are not a substitute for real driver recordings.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\make_tts_samples.ps1
param([string]$OutDir = "samples\tts", [string]$Manifest = "")

Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, [Type]$t) { $task = $asTask.MakeGenericMethod($t).Invoke($null, @($op)); $task.Wait(-1) | Out-Null; $task.Result }

New-Item -ItemType Directory -Force $OutDir | Out-Null
$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$voices = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices

if (-not $Manifest) { $Manifest = Join-Path $PSScriptRoot "tts_samples.json" }
$samples = Get-Content $Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($s in $samples) {
    $voice = $voices | Where-Object { $_.Language -eq $s.voice_lang -and (-not $s.voice -or $_.DisplayName -like "*$($s.voice)*") } | Select-Object -First 1
    if (-not $voice) { Write-Warning "No voice for $($s.voice_lang); skipping $($s.id)"; continue }
    $synth.Voice = $voice
    $stream = Await ($synth.SynthesizeTextToStreamAsync($s.text)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
    $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
    $n = [uint32]$stream.Size
    Await ($reader.LoadAsync($n)) ([uint32]) | Out-Null
    $bytes = New-Object byte[] $n
    $reader.ReadBytes($bytes)
    [IO.File]::WriteAllBytes((Join-Path (Resolve-Path $OutDir) "$($s.id).wav"), $bytes)
    Write-Output "$($s.id).wav  [$($voice.DisplayName)]  $($s.text)"
}
