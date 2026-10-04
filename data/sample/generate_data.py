import pandas as pd
import random
from datetime import datetime, timedelta
random.seed(42)
rows = []
start_date = datetime(2026, 1, 1)

products = {
    "P101": 120,
    "P102": 180,
    "P103": 90,
}
for product_id, base_sales in products.items():
    inventory = 500
    for day in range(120):
        date = start_date + timedelta(days=day)
        price = random.choice([450, 480, 500, 520])
        discount = random.choice([0, 0, 5, 10])
        promotion = 1 if discount >= 5 else 0
        sales = base_sales + random.randint(-20, 20)
        if promotion:
            sales += random.randint(10, 30)
        sales = max(sales, 0)
        inventory = max(inventory - sales, 0)
        if inventory < 150:
            inventory += 400
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "product_id": product_id,
                "sales": sales,
                "price": price,
                "discount": discount,
                "promotion": promotion,
                "inventory": inventory,
            }
        )

df = pd.DataFrame(rows)
df.to_csv("data/sample/sales.csv", index=False)
print("Dataset generated successfully!")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())