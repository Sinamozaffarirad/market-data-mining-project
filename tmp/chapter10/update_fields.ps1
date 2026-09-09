param([string]$Path,[string]$OutputPdf,[string]$InfoPath)
$taskWord = New-Object -ComObject Word.Application
$taskWord.Visible = $false
$taskWord.DisplayAlerts = 0
$taskDoc = $null
try {
    $taskDoc = $taskWord.Documents.Open($Path,$false,$false)
    $taskDoc.Repaginate()
    for ($i=1; $i -le $taskDoc.TablesOfContents.Count; $i++) {
        $taskDoc.TablesOfContents.Item($i).Update()
    }
    $taskDoc.Repaginate()
    for ($i=1; $i -le $taskDoc.TablesOfContents.Count; $i++) {
        $taskDoc.TablesOfContents.Item($i).UpdatePageNumbers()
    }
    $taskDoc.Repaginate()
    $taskStart = $taskDoc.Bookmarks.Item('chapter10_start').Range
    $taskInfo = [ordered]@{
        Pages=$taskDoc.ComputeStatistics(2)
        Chapter10PhysicalPage=$taskStart.Information(3)
        Chapter10PrintedPage=$taskStart.Information(1)
        TablesOfContents=$taskDoc.TablesOfContents.Count
        Footnotes=$taskDoc.Footnotes.Count
        Sections=$taskDoc.Sections.Count
    }
    $taskInfo | ConvertTo-Json | Set-Content -LiteralPath $InfoPath -Encoding utf8
    $taskDoc.Save()
    $taskDoc.ExportAsFixedFormat($OutputPdf,17,$false)
    $taskInfo | ConvertTo-Json
}
finally {
    if ($null -ne $taskDoc) { $taskDoc.Close(0) }
    $taskWord.Quit()
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($taskWord)
}
