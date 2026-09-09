param(
    [Parameter(Mandatory = $true)][string]$Path,
    [string]$QaPdf
)

$wdActiveEndPageNumber = 3
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $null
try {
    $doc = $word.Documents.Open($Path, $false, $true)
    $doc.Repaginate()
    if ($QaPdf) { $doc.ExportAsFixedFormat($QaPdf, 17) }
    "Pages: $($doc.ComputeStatistics(2))"
    "Sections: $($doc.Sections.Count)"
    foreach ($section in $doc.Sections) {
        $start = $section.Range.Duplicate
        $start.Collapse(1)
        "Section $($section.Index) starts on physical page $($start.Information($wdActiveEndPageNumber))"
    }
    $i = 0
    foreach ($p in $doc.Paragraphs) {
        $i++
        $text = $p.Range.Text.Trim([char]13, [char]7, [char]11)
        $page = $p.Range.Information($wdActiveEndPageNumber)
        if ($text -match 'فهرست مطالب|واژه.*اختصار|اختصار.*پرکاربرد' -or ($page -ge 15 -and $page -le 22 -and $text.Length -gt 0)) {
            "Paragraph $i | Page $($p.Range.Information($wdActiveEndPageNumber)) | $text"
        }
    }
}
finally {
    if ($null -ne $doc) { $doc.Close(0) }
    $word.Quit()
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
