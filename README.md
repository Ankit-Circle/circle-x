# price-calculator

## Running with Docker

Build the Docker image:

```bash
docker build -t price-engine .
```

Run the Docker container:

```bash
docker run -p 5000:5000 price-engine
```

## Example API Call

You can test the API using the following `curl` command:

```bash
curl --location '0.0.0.0:5000/process' \
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

