# price-calculator

## Example API Call

You can test the API using the following `curl` command:

```bash
curl --location '0.0.0.0:8000/process/' \
--header 'Content-Type: application/json' \
--data '{
    "Category": "Audio",
    "Sub_Category": "Bluetooth Speaker",
    "Product_Age_Years1": 9,
    "brand_tier": "Premium",
    "warranty_period": "6 months",
    "Condition_Tier": "Very Good",
    "Functional_Status": "Fully Functional",
    "Best_Online_Price": 4500
}'
```