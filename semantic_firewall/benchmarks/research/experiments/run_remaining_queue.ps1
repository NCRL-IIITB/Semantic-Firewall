$ErrorActionPreference = "Stop"

$scripts = @(
    "semantic_firewall\benchmarks\research\experiments\15_multidataset_eval.py",
    "semantic_firewall\benchmarks\research\experiments\05_unsafe_scaled.py --num-samples 2000",
    "semantic_firewall\benchmarks\research\experiments\07d_ablation_config9.py",
    "semantic_firewall\benchmarks\research\experiments\17_clean_latency.py"
)

foreach ($script in $scripts) {
    Write-Host "`n======================================="
    Write-Host "Executing $script ..."
    Write-Host "======================================="
    
    # Split to handle arguments like --num-samples 2000
    $parts = $script -split ' '
    $file = $parts[0]
    $args = $parts[1..($parts.Length-1)]
    
    python $file $args
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Script $script failed with exit code $LASTEXITCODE."
        exit $LASTEXITCODE
    }
}

Write-Host "ALL REMAINING SCRIPTS COMPLETED SUCCESSFULLY!"
