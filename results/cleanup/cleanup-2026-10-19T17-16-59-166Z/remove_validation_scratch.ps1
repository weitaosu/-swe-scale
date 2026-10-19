$ErrorActionPreference = 'Stop'
$plan = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'post_validation_cleanup_plan.json') -Raw | ConvertFrom-Json
$root = [System.IO.Path]::GetFullPath([string]$plan.project_root).TrimEnd('\')
$prefix = $root + [System.IO.Path]::DirectorySeparatorChar
$validated = @()
foreach ($target in $plan.targets) {
    $absolute = [System.IO.Path]::GetFullPath((Join-Path $root ([string]$target.path)))
    if (-not $absolute.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Target outside workspace: $absolute" }
    if (-not (Test-Path -LiteralPath $absolute)) { continue }
    $item = Get-Item -LiteralPath $absolute -Force
    $resolved = (Resolve-Path -LiteralPath $absolute).ProviderPath
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Resolved target outside workspace: $resolved" }
    $cursor = $item
    while ($cursor.FullName -ne $root) {
        if (($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse point refused: $($cursor.FullName)" }
        $cursor = Get-Item -LiteralPath (Split-Path -Parent $cursor.FullName) -Force
    }
    if ($target.kind -eq 'directory' -and -not $item.PSIsContainer) { throw "Expected directory: $absolute" }
    if ($target.kind -eq 'file') {
        if ($item.PSIsContainer) { throw "Expected file: $absolute" }
        $actualHash = (Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $target.expected_sha256) { throw "File changed since review; preserved: $absolute" }
    }
    $validated += [PSCustomObject]@{path=$target.path; absolute=$resolved; kind=$target.kind; reason=$target.reason}
}
$removed = @()
try {
    foreach ($target in $validated) {
        if ($target.kind -eq 'directory') { Remove-Item -LiteralPath $target.absolute -Recurse -Force }
        else { Remove-Item -LiteralPath $target.absolute -Force }
        $removed += [PSCustomObject]@{path=$target.path; kind=$target.kind; reason=$target.reason; removed=$true}
        Write-Output ("Removed: " + $target.path)
    }
}
finally {
    ConvertTo-Json -InputObject $removed -Depth 5 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'post_validation_removal_records.json') -Encoding UTF8
}
