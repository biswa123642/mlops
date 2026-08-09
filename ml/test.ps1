$scoringUri = "https://ml.altudo.net/api/v1/endpoint/customer-churn-endpoint/score"
$primaryKey = "2mi7ZIDiVVOv0kcSJK0qbZvUTNGpLm2DHyHdLQLEKiVl79iUrG0kJQQJ99CHAAAAAAAAAAAAINFRAZML1eS8"

$headers = @{
    "Content-Type" = "application/json"
    "Authorization" = "Bearer $primaryKey"
}

$body = Get-Content `
    -Path .\test_request.json `
    -Raw

$response = Invoke-RestMethod `
    -Method Post `
    -Uri $scoringUri `
    -Headers $headers `
    -Body $body

$response | ConvertTo-Json -Depth 10