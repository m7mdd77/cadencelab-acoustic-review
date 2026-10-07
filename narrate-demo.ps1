param([Parameter(Mandatory=$true)][string]$TextFile,
      [Parameter(Mandatory=$true)][string]$OutputFile)
$ErrorActionPreference = 'Stop'
$voice = New-Object -ComObject SAPI.SpVoice
$stream = New-Object -ComObject SAPI.SpFileStream
try {
    $stream.Format.Type = 22
    $stream.Open([System.IO.Path]::GetFullPath($OutputFile), 3, $false)
    $voice.AudioOutputStream = $stream
    $voice.Rate = -1
    $null = $voice.Speak([System.IO.File]::ReadAllText([System.IO.Path]::GetFullPath($TextFile)))
} finally {
    $stream.Close()
}
