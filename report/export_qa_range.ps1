param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Output
)

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $null
try {
    $doc = $word.Documents.Open($Path, $false, $true)
    $doc.Repaginate()
    # wdExportFromTo = 3; export only the revised front-matter pages.
    $doc.ExportAsFixedFormat($Output, 17, $false, 0, 3, 6, 27)
}
finally {
    if ($null -ne $doc) { $doc.Close(0) }
    $word.Quit()
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
