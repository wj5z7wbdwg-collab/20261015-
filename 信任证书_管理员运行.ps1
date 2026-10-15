# 一键信任猜拳游戏自签名证书
# 使用方法：右键本文件 -> “使用 PowerShell 运行”；若提示权限请选择“以管理员身份运行”
$ErrorActionPreference = 'Stop'
$certPath = Join-Path $PSScriptRoot 'cert.pem'

Write-Host "正在导入证书: $certPath" -ForegroundColor Cyan
try {
    Import-Certificate -FilePath $certPath -CertStoreLocation Cert:\LocalMachine\Root | Out-Null
    Write-Host "[成功] 证书已加入“受信任的根证书颁发机构”。" -ForegroundColor Green
    Write-Host "请完全关闭浏览器后重新打开 https://10.10.236.12:8000 ，红色“不安全”警告即会消失。" -ForegroundColor Green
} catch {
    Write-Host "[失败] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "请右键“开始”按钮 -> Windows PowerShell(管理员) / 终端(管理员)，然后执行：" -ForegroundColor Yellow
    Write-Host "  Import-Certificate -FilePath `"$certPath`" -CertStoreLocation Cert:\LocalMachine\Root" -ForegroundColor Yellow
}
Write-Host ""
Read-Host "按回车键退出"
