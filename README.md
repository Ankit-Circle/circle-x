# circle-x AI Api server

## Running with Docker

Build the Docker image:

```bash
docker build -t circle-x .
```

Run the Docker container:

```bash
docker run -p 5000:5000 circle-x
```

## Example API Call

You can test the pricing API using the following `curl` command:

```bash
curl --location '0.0.0.0:5000/api/pricing' \
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

You can test the Image enhancement API using

```bash
curl --location '0.0.0.0:5000/api/enhance' \
--header 'Content-Type: application/json' \
--data '{
    "image_url": "https://res.cloudinary.com/dkjsiqjfr/image/upload/v1752215572/product_submissions/IMG_5593_1.jpg"
}'
```

