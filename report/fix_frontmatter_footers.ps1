param(
    [Parameter(Mandatory = $true)][string]$Source,
    [Parameter(Mandatory = $true)][string]$Output,
    [Parameter(Mandatory = $true)][string]$QaPdf
)

$wdCollapseStart = 1
$wdSectionBreakNextPage = 2
$wdActiveEndPageNumber = 3
$wdExportFormatPDF = 17

if (Test-Path -LiteralPath $Output) {
    throw "Output already exists: $Output"
}

Copy-Item -LiteralPath $Source -Destination $Output
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $null
try {
    $doc = $word.Documents.Open($Output, $false, $false)
    $doc.Repaginate()

    # The TOC content control has been unwrapped beforehand so Word can place
    # section breaks between its individual physical pages.
    for ($page = 16; $page -ge 7; $page--) {
        $pageStart = $doc.GoTo(1, 1, $page)
        if ($pageStart.Information($wdActiveEndPageNumber) -ne $page) {
            throw "Could not resolve the start of physical page $page."
        }
        $word.Selection.SetRange($pageStart.Start, $pageStart.Start)
        $word.Selection.InsertBreak($wdSectionBreakNextPage)
        $doc.Repaginate()
    }

    $footerWords = @(
        'یک', 'دو', 'سه', 'چهار', 'پنج', 'شش', 'هفت', 'هشت', 'نه', 'ده', 'یازده',
        'دوازده', 'سیزده', 'چهارده', 'پانزده', 'شانزده', 'هفده', 'هجده', 'نوزده', 'بیست',
        'بیست و یک', 'بیست و دو'
    )
    $found = @{}
    foreach ($section in $doc.Sections) {
        $start = $section.Range.Duplicate
        $start.Collapse($wdCollapseStart)
        $page = $start.Information($wdActiveEndPageNumber)
        if ($page -ge 6 -and $page -le 27) {
            $footer = $section.Footers.Item(1)
            $footer.LinkToPrevious = $false
            $footer.Range.Text = $footerWords[$page - 6]
            $found[$page] = $true
        }
    }
    for ($page = 6; $page -le 27; $page++) {
        if (-not $found.ContainsKey($page)) {
            throw "No section begins on physical page $page after inserting the Table of Contents section break."
        }
    }

    $doc.Repaginate()
    $doc.Save()
    $doc.ExportAsFixedFormat($QaPdf, $wdExportFormatPDF)
    "Pages after edit: $($doc.ComputeStatistics(2))"
    foreach ($section in $doc.Sections) {
        $start = $section.Range.Duplicate
        $start.Collapse($wdCollapseStart)
        $page = $start.Information($wdActiveEndPageNumber)
        if ($page -ge 6 -and $page -le 27) {
            "Section $($section.Index) | physical page $page | footer $($section.Footers.Item(1).Range.Text.Trim([char]13, [char]7))"
        }
    }
}
finally {
    if ($null -ne $doc) { $doc.Close(0) }
    $word.Quit()
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
