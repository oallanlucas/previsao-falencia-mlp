# Executa os notebooks 01 a 06 em ordem, salvando as saídas dentro de cada notebook.
# Uso:  powershell -ExecutionPolicy Bypass -File rodar_tudo.ps1            (execução completa)
#       powershell -ExecutionPolicy Bypass -File rodar_tudo.ps1 -Rapido    (teste rápido do código)
# Parâmetros opcionais: -Python (executável com torch/sklearn), -Kernel (nome do kernel do Jupyter), -Inicio (nº do
# primeiro notebook, para retomar depois de uma interrupção; o cache da MLP evita refazer o que já foi calculado).
param(
    [switch]$Rapido,
    [string]$Python = "python",
    [string]$Kernel = "python3",
    [int]$Inicio = 1
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if ($Rapido) { $env:RAPIDO = "1" } else { Remove-Item Env:RAPIDO -ErrorAction SilentlyContinue }
$env:PYTHONIOENCODING = "utf-8"
# Nesta máquina: o Jupyter está no Python 3.13 e o torch com CUDA no 3.10. O kernel "py310cuda" (em .jupyter_local)
# aponta para o 3.10. Em outra máquina, basta um Python com jupyter + requirements.txt e os parâmetros padrão.
if (Test-Path ".jupyter_local\kernels") { $env:JUPYTER_PATH = (Resolve-Path ".jupyter_local").Path }
New-Item -ItemType Directory -Force logs | Out-Null
# O 02 roda de novo no fim: a seção 9.4 dele usa as previsões finais salvas pelos notebooks 03 e 04.
$notebooks = @("01_leitura_dados", "02_eda_estabilidade", "03_regressao_logistica", "04_mlp",
               "05_comparacao_final", "06_alerta_horizonte_vies_tempo", "02_eda_estabilidade")
for ($i = 0; $i -lt $notebooks.Count; $i++) {
    $nb = $notebooks[$i]
    if ($i -lt 6 -and [int]$nb.Substring(0, 2) -lt $Inicio) { continue }
    $t0 = Get-Date
    Write-Host "$(Get-Date -Format 'HH:mm:ss')  rodando $nb ..."
    # O nbconvert escreve o progresso no stderr; no PowerShell 5.1, com "Stop", isso vira erro fatal. O sucesso é
    # verificado pelo código de saída logo abaixo.
    $ErrorActionPreference = "Continue"
    & $Python -m jupyter nbconvert --to notebook --execute --inplace `
        --ExecutePreprocessor.kernel_name=$Kernel --ExecutePreprocessor.timeout=-1 "$nb.ipynb" *> "logs\$nb.txt"
    $ErrorActionPreference = "Stop"
    if ($LASTEXITCODE -ne 0) { Write-Host "ERRO em $nb (veja logs\$nb.txt)"; exit 1 }
    Write-Host ("    ok em {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes)
}
Write-Host "concluído"
