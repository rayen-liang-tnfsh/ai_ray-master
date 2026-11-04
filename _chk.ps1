$j = Get-Content -LiteralPath "線性回歸\高麗菜估價.ipynb" -Raw -Encoding UTF8 | ConvertFrom-Json
$out = New-Object System.Collections.Generic.List[string]
for ($i = 0; $i -lt $j.cells.Count; $i++) {
    $s = $j.cells[$i].source -join ""
    $lines = $s -split "`n"
    $out.Add("===== CELL $($i+1) type=$($j.cells[$i].cell_type) chars=$($s.Length) lines=$($lines.Count) =====")
    $out.Add("FIRST: " + $lines[0])
    $out.Add("LAST : " + $lines[$lines.Count - 1])
}
$out | Set-Content -LiteralPath "_chk.txt" -Encoding UTF8
DONE
