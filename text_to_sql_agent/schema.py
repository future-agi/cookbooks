"""Database schema for the Text to SQL Agent."""

import datetime
import random
import uuid

# Database schema for e-commerce platform
ECOMMERCE_SCHEMA = """
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    signup_date DATE,
    location_country TEXT,
    location_state TEXT,
    device_type TEXT,
    preferred_category TEXT
);

CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT,
    session_start_time TIMESTAMP,
    session_end_time TIMESTAMP,
    device_type TEXT,
    browser TEXT,
    referral_source TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    name TEXT,
    category TEXT,
    brand TEXT,
    price FLOAT,
    inventory_level INTEGER,
    rating FLOAT
);

CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    user_id TEXT,
    session_id TEXT,
    order_date DATE,
    order_total FLOAT,
    payment_method TEXT,
    shipping_cost FLOAT,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE order_items (
    order_item_id TEXT PRIMARY KEY,
    order_id TEXT,
    product_id TEXT,
    quantity INTEGER,
    price_each FLOAT,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE page_views (
    view_id TEXT PRIMARY KEY,
    session_id TEXT,
    timestamp TIMESTAMP,
    product_id TEXT,
    event_type TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE inventory_log (
    product_id TEXT,
    date DATE,
    stock_level INTEGER,
    restocked_amount INTEGER,
    PRIMARY KEY (product_id, date),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
"""


def generate_user_id():
    """Generates a unique user ID."""
    return f"user_{str(uuid.uuid4())[:8]}"

def generate_session_id():
    """Generates a unique session ID."""
    return f"sess_{str(uuid.uuid4())[:8]}"

def generate_product_id():
    """Generates a unique product ID."""
    return f"prod_{str(uuid.uuid4())[:8]}"

def generate_product_name(brand, category):
    """Generates a unique product name based on the brand and category."""
    adjectives = ["Premium", "Elite", "Classic", "Advanced", "Ultra", "Essential", "Deluxe", "Professional", "Ultimate", "Compact"]

    category_descriptors = {
        "Electronics": ["Wireless", "Smart", "Digital", "HD", "Bluetooth", "4K", "Portable", "Fast-Charging"],
        "Clothing": ["Slim-Fit", "Comfort", "Athletic", "Vintage", "Signature", "Performance", "Relaxed", "Urban"],
        "Home & Garden": ["Ergonomic", "Modern", "Rustic", "Sustainable", "Handcrafted", "Space-Saving"],
        "Books": ["Bestselling", "Award-Winning", "Illustrated", "Comprehensive", "Pocket", "Collector's Edition"],
        "Toys": ["Interactive", "Educational", "Wooden", "Remote-Control", "Collectible", "Plush"],
        "Sports": ["Pro", "Lightweight", "All-Season", "Endurance", "Competition", "Training"],
        "Beauty": ["Organic", "Hydrating", "Rejuvenating", "Natural", "Anti-Aging", "SPF"],
        "Furniture": ["Modular", "Convertible", "Scandinavian", "Memory Foam", "Adjustable", "Reclining"],
        "Jewelry": ["Handcrafted", "Sterling", "14K Gold", "Vintage", "Statement", "Minimalist"],
        "Automotive": ["Heavy-Duty", "All-Weather", "Premium", "Synthetic", "Universal Fit"],
        "Food & Grocery": ["Organic", "Gluten-Free", "Artisanal", "Whole Grain", "Sugar-Free", "Free-Range"],
        "Health & Wellness": ["Natural", "Clinical Strength", "Essential", "Daily", "Maximum Strength"],
        "Pet Supplies": ["Grain-Free", "Orthopedic", "Durable", "Hypoallergenic", "Automatic"],
        "Office Supplies": ["Ergonomic", "Recycled", "Refillable", "Wireless", "Heavy-Duty"],
        "Art & Crafts": ["Professional", "Washable", "Non-Toxic", "Artist-Grade", "Beginner"]
    }

    category_items = {
        "Electronics": ["Smartphone", "Laptop", "Tablet", "Headphones", "Smartwatch", "Speaker", "Camera", "Power Bank", "Monitor", "Router"],
        "Clothing": ["T-Shirt", "Jeans", "Jacket", "Sneakers", "Dress", "Sweater", "Hoodie", "Shorts", "Socks", "Hat"],
        "Home & Garden": ["Coffee Table", "Chair", "Lamp", "Planter", "Throw Pillow", "Comforter", "Cookware Set", "Blender"],
        "Books": ["Novel", "Cookbook", "Biography", "Self-Help Book", "Travel Guide", "History Book", "Children's Book"],
        "Toys": ["Building Set", "Doll", "Action Figure", "Board Game", "Puzzle", "Stuffed Animal", "Robot", "Art Kit"],
        "Sports": ["Running Shoes", "Yoga Mat", "Dumbbell Set", "Tennis Racket", "Basketball", "Water Bottle", "Fitness Tracker"],
        "Beauty": ["Face Cream", "Shampoo", "Lipstick", "Foundation", "Serum", "Perfume", "Face Mask", "Moisturizer"],
        "Furniture": ["Sofa", "Bed Frame", "Dining Table", "Bookshelf", "Desk", "Nightstand", "Armchair", "TV Stand"],
        "Jewelry": ["Necklace", "Earrings", "Bracelet", "Ring", "Watch", "Anklet", "Cufflinks", "Pendant"],
        "Automotive": ["Floor Mats", "Phone Mount", "Oil Filter", "Dash Cam", "Wiper Blades", "Car Freshener"],
        "Food & Grocery": ["Coffee", "Chocolate", "Pasta", "Snack Mix", "Protein Bars", "Olive Oil", "Granola", "Tea"],
        "Health & Wellness": ["Vitamin", "Protein Powder", "Essential Oil", "Massager", "Fitness Tracker", "Thermometer"],
        "Pet Supplies": ["Dog Bed", "Cat Toy", "Pet Food", "Scratching Post", "Collar", "Fish Tank", "Pet Carrier"],
        "Office Supplies": ["Notebook", "Desk Organizer", "Pen Set", "Stapler", "Whiteboard", "Desk Lamp", "File Cabinet"],
        "Art & Crafts": ["Paint Set", "Sketchbook", "Yarn", "Canvas", "Brush Set", "Colored Pencils", "Clay Kit"]
    }

    descriptors = category_descriptors.get(category, category_descriptors[list(category_descriptors.keys())[0]])
    items = category_items.get(category, category_items[list(category_items.keys())[0]])

    adjective = random.choice(adjectives)
    descriptor = random.choice(descriptors)
    item = random.choice(items)

    name_patterns = [
        f"{brand} {descriptor} {item}",
        f"{brand} {adjective} {item}",
        f"{brand} {item} {random.randint(1, 1000)}",
        f"{adjective} {item} by {brand}",
        f"{descriptor} {item} - {brand}",
        f"{brand} {descriptor} {adjective} {item}"
    ]

    return random.choice(name_patterns)

def generate_order_id():
    """Generates a unique order ID."""
    return f"ord_{str(uuid.uuid4())[:8]}"

def generate_view_id():
    """Generates a unique view ID."""
    return f"view_{str(uuid.uuid4())[:8]}"

def generate_order_item_id():
    """Generates a unique order item ID."""
    return f"item_{str(uuid.uuid4())[:8]}"

def generate_sample_data(num_users=50, num_products=100, days_back=30):
    """Generates sample data for the database."""
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=days_back)

    countries = ["USA", "Canada", "UK", "Germany", "Australia", "Japan", "France", "Spain", "Italy", "Brazil", "China", "India", "South Korea", "Mexico", "Netherlands"]
    states = ["CA", "NY", "TX", "FL", "IL", "WA", "MA", "CO", "OR", "GA", "ON", "BC", "QC", "London", "Manchester", "Liverpool", "Berlin", "Munich", "Hamburg", "Paris", "Lyon", "Madrid", "Barcelona", "Sydney", "Melbourne"]
    device_types = ["Desktop", "Mobile", "Tablet", "Smart TV", "Gaming Console", "Wearable", "Smart Speaker"]
    browsers = ["Chrome", "Firefox", "Safari", "Edge", "Opera", "Brave", "Vivaldi", "Samsung Internet", "Chrome Mobile", "Safari Mobile"]
    referral_sources = ["Direct", "Google", "Facebook", "Instagram", "Email", "Twitter", "Pinterest", "YouTube", "TikTok", "LinkedIn", "Reddit", "Affiliate", "Bing", "Organic Search", "Paid Search", "Retargeting Campaign"]
    categories = ["Electronics", "Clothing", "Home & Garden", "Books", "Toys", "Sports", "Beauty", "Furniture", "Jewelry", "Automotive", "Food & Grocery", "Health & Wellness", "Pet Supplies", "Office Supplies", "Art & Crafts"]
    brands = [
        # Electronics
        "Apple", "Samsung", "Sony", "LG", "Dell", "HP", "Lenovo", "Asus", "Microsoft", "Google", "Bose", "Sonos", "Dyson", "Philips", "Canon", "GoPro",
        # Clothing
        "Nike", "Adidas", "Zara", "H&M", "Levi's", "Under Armour", "The North Face", "Gucci", "Louis Vuitton", "Uniqlo", "Patagonia",
        # Home & Garden
        "IKEA", "Wayfair", "Crate & Barrel", "Williams-Sonoma", "KitchenAid", "Breville", "Cuisinart", "OXO",
        # Beauty
        "L'Oreal", "Estée Lauder", "Sephora", "MAC Cosmetics", "Kiehl's", "Fenty Beauty", "Glossier",
        # Toys
        "LEGO", "Hasbro", "Mattel", "Fisher-Price", "Nintendo",
        # Other
        "Amazon Basics", "Whole Foods", "Trader Joe's", "Costco Wholesale"
    ]
    payment_methods = ["Credit Card", "PayPal", "Apple Pay", "Google Pay", "Bank Transfer", "Venmo", "Bitcoin", "Klarna", "Affirm", "AfterPay", "Store Credit", "Gift Card"]
    event_types = ["view", "add_to_cart", "remove_from_cart", "checkout_click", "purchase", "add_to_wishlist", "product_click", "filter_apply", "search_query", "zoom_image", "read_reviews", "video_play"]

    users = []
    user_ids = []
    for _ in range(num_users):
        user_id = generate_user_id()
        user_ids.append(user_id)
        signup_date = start_date + datetime.timedelta(days=random.randint(0, days_back))
        users.append({
            "user_id": user_id,
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "location_country": random.choice(countries),
            "location_state": random.choice(states),
            "device_type": random.choice(device_types),
            "preferred_category": random.choice(categories)
        })

    products = []
    product_ids = []
    for _ in range(num_products):
        product_id = generate_product_id()
        product_ids.append(product_id)
        category = random.choice(categories)
        brand = random.choice(brands)

        base_price = {
            "Electronics": (49.99, 1299.99),
            "Clothing": (14.99, 199.99),
            "Home & Garden": (19.99, 599.99),
            "Books": (7.99, 39.99),
            "Toys": (9.99, 99.99),
            "Sports": (12.99, 249.99),
            "Beauty": (6.99, 129.99),
            "Furniture": (49.99, 1999.99),
            "Jewelry": (19.99, 2499.99),
            "Automotive": (9.99, 299.99),
            "Food & Grocery": (3.99, 49.99),
            "Health & Wellness": (8.99, 89.99),
            "Pet Supplies": (5.99, 149.99),
            "Office Supplies": (4.99, 199.99),
            "Art & Crafts": (5.99, 79.99)
        }.get(category, (9.99, 199.99))

        price = round(random.uniform(base_price[0], base_price[1]), 2)

        price = round(price - 0.01, 2) if random.random() < 0.7 else round(price - 0.05, 2)

        if category in ["Electronics", "Furniture", "Jewelry"]:
            inventory = random.randint(5, 150)
        else:
            inventory = random.randint(20, 500)

        rating_weights = [0.01, 0.04, 0.15, 0.35, 0.45]
        rating = random.choices([1.0, 2.0, 3.0, 4.0, 5.0], weights=rating_weights)[0]
        if rating < 5.0:
            rating += random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5])
            rating = round(rating, 1)

        product_name = generate_product_name(brand, category)

        products.append({
            "product_id": product_id,
            "name": product_name,
            "category": category,
            "brand": brand,
            "price": price,
            "inventory_level": inventory,
            "rating": rating
        })

    sessions = []
    session_ids = []
    for user_id in user_ids:
        user_preferred_devices = random.sample(device_types, min(3, len(device_types)))
        user_preferred_browsers = random.sample(browsers, min(2, len(browsers)))
        user_preferred_sources = random.sample(referral_sources, min(4, len(referral_sources)))

        engagement_level = random.choices(["low", "medium", "high"], weights=[0.3, 0.5, 0.2])[0]
        num_sessions = {
            "low": random.randint(1, 3),
            "medium": random.randint(3, 8),
            "high": random.randint(8, 15)
        }[engagement_level]

        for _ in range(num_sessions):
            session_id = generate_session_id()
            session_ids.append(session_id)

            session_date = start_date + datetime.timedelta(days=random.randint(0, days_back))
            weekday = session_date.weekday()
            is_weekend = weekday >= 5

            if is_weekend:
                peak_hours = [(10, 13), (14, 17)]
                weights = [0.15, 0.4, 0.3, 0.15]
                hour_segment = random.choices([0, 1, 2, 3], weights=weights)[0]
                if hour_segment == 0:
                    start_hour = random.randint(7, 9)
                elif hour_segment == 1:
                    start_hour = random.randint(*peak_hours[0])
                elif hour_segment == 2:
                    start_hour = random.randint(*peak_hours[1])
                else:
                    start_hour = random.randint(18, 23)
            else:
                peak_hours = [(7, 9), (12, 13), (17, 22)]
                weights = [0.2, 0.1, 0.1, 0.6]
                hour_segment = random.choices([0, 1, 2, 3], weights=weights)[0]
                if hour_segment == 0:
                    start_hour = random.randint(*peak_hours[0])
                elif hour_segment == 1:
                    start_hour = random.randint(*peak_hours[1])
                elif hour_segment == 2:
                    start_hour = random.randint(*peak_hours[2])
                else:
                    excluded_hours = list(range(peak_hours[0][0], peak_hours[0][1] + 1))
                    excluded_hours.extend(list(range(peak_hours[1][0], peak_hours[1][1] + 1)))
                    excluded_hours.extend(list(range(peak_hours[2][0], peak_hours[2][1] + 1)))
                    available_hours = [h for h in range(6, 24) if h not in excluded_hours]
                    start_hour = random.choice(available_hours) if available_hours else random.randint(6, 24)

            start_minute = random.randint(0, 59)
            session_start = datetime.datetime(
                session_date.year, session_date.month, session_date.day,
                start_hour, start_minute
            )

            device_type = random.choices(user_preferred_devices, weights=[0.7, 0.2, 0.1])[0] if user_preferred_devices else random.choice(device_types)

            if device_type == "Mobile":
                session_length = datetime.timedelta(minutes=random.randint(3, 25))
            elif device_type == "Tablet":
                session_length = datetime.timedelta(minutes=random.randint(10, 45))
            elif device_type == "Smart TV" or device_type == "Gaming Console":
                session_length = datetime.timedelta(minutes=random.randint(20, 90))
            else:
                session_length = datetime.timedelta(minutes=random.randint(8, 60))

            if 19 <= start_hour <= 23:
                session_length = session_length * random.uniform(1.2, 1.5)

            session_end = session_start + session_length

            browser = random.choices(user_preferred_browsers, weights=[0.8, 0.2])[0] if user_preferred_browsers else random.choice(browsers)

            referral_weights = [0.5, 0.5]
            is_direct = random.choices([True, False], weights=referral_weights)[0]
            referral_source = "Direct" if is_direct else random.choice(user_preferred_sources)

            sessions.append({
                "session_id": session_id,
                "user_id": user_id,
                "session_start_time": session_start.strftime("%Y-%m-%d %H:%M:%S"),
                "session_end_time": session_end.strftime("%Y-%m-%d %H:%M:%S"),
                "device_type": device_type,
                "browser": browser,
                "referral_source": referral_source
            })

    page_views = []
    for session_id in session_ids:
        for _ in range(random.randint(1, 20)):
            session = next(s for s in sessions if s["session_id"] == session_id)
            start_time = datetime.datetime.strptime(session["session_start_time"], "%Y-%m-%d %H:%M:%S")
            end_time = datetime.datetime.strptime(session["session_end_time"], "%Y-%m-%d %H:%M:%S")
            timestamp = start_time + (end_time - start_time) * random.random()
            product_id = random.choice(product_ids)
            page_views.append({
                "view_id": generate_view_id(),
                "session_id": session_id,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "product_id": product_id,
                "event_type": random.choice(event_types)
            })

    orders = []
    order_ids = []

    seasonal_factors = {}

    for day in range(days_back + 1):
        date = today - datetime.timedelta(days=day)
        seasonal_factors[date.strftime('%Y-%m-%d')] = 1.0

    for day in range(7):
        if days_back >= day:
            date = today - datetime.timedelta(days=day)
            seasonal_factors[date.strftime('%Y-%m-%d')] *= 1.5

    for day in range(days_back + 1):
        date = today - datetime.timedelta(days=day)
        if date.weekday() >= 5:
            seasonal_factors[date.strftime('%Y-%m-%d')] *= 1.3

    sale_days = random.sample(range(days_back + 1), 3)
    for day in sale_days:
        date = today - datetime.timedelta(days=day)
        seasonal_factors[date.strftime('%Y-%m-%d')] *= 2.0

    for user_id in user_ids:
        purchase_likelihood = random.choices(
            ["non-buyer", "occasional", "regular", "frequent"],
            weights=[0.2, 0.4, 0.3, 0.1]
        )[0]

        purchase_prob = {
            "non-buyer": 0.1,
            "occasional": 0.5,
            "regular": 0.8,
            "frequent": 0.95
        }[purchase_likelihood]

        if random.random() < purchase_prob:
            max_orders = {
                "non-buyer": 1,
                "occasional": 2,
                "regular": 4,
                "frequent": 8
            }[purchase_likelihood]

            num_orders = min(random.randint(1, max_orders), len([s for s in sessions if s["user_id"] == user_id]))

            user_sessions = [s for s in sessions if s["user_id"] == user_id]
            if not user_sessions:
                continue

            user_sessions.sort(key=lambda s: s["session_end_time"])

            if purchase_likelihood == "non-buyer" or purchase_likelihood == "occasional":
                session_purchase_weights = [0.1 + (i * 0.15) for i in range(len(user_sessions))]
            else:
                session_purchase_weights = [0.5 + (i * 0.05) for i in range(len(user_sessions))]

            session_purchase_weights = [min(w, 1.0) for w in session_purchase_weights]

            order_count = 0
            for i, session in enumerate(user_sessions):
                if order_count >= num_orders:
                    break

                session_date = datetime.datetime.strptime(session["session_end_time"], "%Y-%m-%d %H:%M:%S").date()
                date_str = session_date.strftime('%Y-%m-%d')
                seasonal_boost = seasonal_factors.get(date_str, 1.0)

                adjusted_prob = min(session_purchase_weights[i] * seasonal_boost, 1.0)

                if random.random() < adjusted_prob:
                    order_id = generate_order_id()
                    order_ids.append(order_id)

                    base_shipping = 4.99
                    will_be_free = random.random() < 0.7

                    if will_be_free:
                        shipping_cost = 0.0
                    else:
                        weight_factor = random.uniform(0.8, 1.5)
                        shipping_cost = round(base_shipping * weight_factor, 2)

                    payment_weights = [0.65, 0.2, 0.05, 0.05, 0.05]
                    payment_method = random.choices(payment_methods[:5], weights=payment_weights)[0]

                    orders.append({
                        "order_id": order_id,
                        "user_id": user_id,
                        "session_id": session["session_id"],
                        "order_date": session_date.strftime("%Y-%m-%d"),
                        "order_total": 0,  # Placeholder
                        "payment_method": payment_method,
                        "shipping_cost": shipping_cost
                    })

                    order_count += 1

    order_items = []
    for order_id in order_ids:
        order_total = 0
        for _ in range(random.randint(1, 5)):
            product = random.choice(products)
            quantity = random.randint(1, 5)
            price_each = product["price"]
            item_total = quantity * price_each
            order_total += item_total
            order_items.append({
                "order_item_id": generate_order_item_id(),
                "order_id": order_id,
                "product_id": product["product_id"],
                "quantity": quantity,
                "price_each": price_each
            })

        for order in orders:
            if order["order_id"] == order_id:
                order["order_total"] = round(order_total + order["shipping_cost"], 2)
                break

    inventory_log = []

    product_popularity = {product_id: 0 for product_id in product_ids}

    for item in order_items:
        product_id = item["product_id"]
        product_popularity[product_id] = product_popularity.get(product_id, 0) + item["quantity"]

    if product_popularity:
        max_popularity = max(product_popularity.values()) if product_popularity.values() else 1
        if max_popularity > 0:
            for product_id in product_popularity:
                product_popularity[product_id] = product_popularity[product_id] / max_popularity

    restock_schedules = {}
    for product in products:
        product_id = product["product_id"]
        category = product["category"]
        price = product["price"]
        popularity = product_popularity.get(product_id, 0)

        if category in ["Food & Grocery", "Health & Wellness"]:
            restock_frequency = max(3, int(14 - (10 * popularity)))
        elif category in ["Electronics", "Furniture"]:
            restock_frequency = max(7, int(21 - (7 * popularity)))
        else:
            restock_frequency = max(5, int(18 - (9 * popularity)))

        if price > 500:
            base_restock = random.randint(5, 20)
        elif price > 100:
            base_restock = random.randint(10, 50)
        else:
            base_restock = random.randint(20, 100)

        restock_amount = int(base_restock * (0.5 + (popularity * 1.5)))

        restock_schedules[product_id] = {
            "frequency": restock_frequency,
            "amount": restock_amount,
            "variance": 0.3,
            "last_restock": random.randint(0, restock_frequency)
        }

    for product_id in product_ids:
        product = next(p for p in products if p["product_id"] == product_id)
        current_stock = product["inventory_level"]
        category = product["category"]
        schedule = restock_schedules[product_id]
        days_since_restock = schedule["last_restock"]
        popularity = product_popularity.get(product_id, 0.1)

        critical_threshold = max(5, int(10 * popularity))

        for day in range(days_back, -1, -1):
            log_date = today - datetime.timedelta(days=day)
            weekday = log_date.weekday()
            restocked = 0

            scheduled_restock = days_since_restock >= schedule["frequency"]

            emergency_restock = current_stock <= critical_threshold

            is_business_day = weekday < 5

            restock_happens = False

            if emergency_restock and is_business_day:
                restock_happens = random.random() < 0.9
            elif scheduled_restock and is_business_day:
                restock_happens = random.random() < 0.8
            elif scheduled_restock:
                restock_happens = random.random() < 0.3
            elif emergency_restock:
                restock_happens = random.random() < 0.5

            if restock_happens:
                base_amount = schedule["amount"]

                variance_factor = random.uniform(1 - schedule["variance"], 1 + schedule["variance"])

                if emergency_restock and not scheduled_restock:
                    emergency_factor = random.uniform(0.3, 0.7)
                    restocked = max(5, int(base_amount * variance_factor * emergency_factor))
                else:
                    restocked = max(5, int(base_amount * variance_factor))

                current_stock += restocked

                days_since_restock = 0
            else:
                days_since_restock += 1

            sale_probability = 0.3 + (0.5 * popularity)

            if weekday >= 5:
                sale_probability *= 1.3

            if current_stock > 0 and random.random() < sale_probability:
                base_sales = 1 + int(8 * popularity)

                sales_variance = random.uniform(0.7, 1.3)

                sold = min(int(base_sales * sales_variance), current_stock)
                current_stock -= sold

            inventory_log.append({
                "product_id": product_id,
                "date": log_date.strftime("%Y-%m-%d"),
                "stock_level": current_stock,
                "restocked_amount": restocked
            })

    return {
        "users": users,
        "products": products,
        "sessions": sessions,
        "orders": orders,
        "order_items": order_items,
        "page_views": page_views,
        "inventory_log": inventory_log
    }

SAMPLE_DATA = generate_sample_data(num_users=20, num_products=30, days_back=30)

# EXAMPLE_QUERIES = [
#     # Time Series & Trends
#     "Which product had the most views over the last 7 days?",
#     "What was the total revenue for each day last month?",
#     "How many sessions occurred per hour yesterday?",

#     # Behavioural Analysis
#     "Which category has the highest add-to-cart rate?",
#     "How many users placed an order within 7 days of signup?",
#     "What is the most popular device type by sales volume?",
#     "List users who abandoned carts but didn't buy.",

#     # Sales & Product Performance
#     "Top 5 products by quantity sold this quarter?",
#     "Compare average order size by payment method.",
#     "What's the average rating of products sold in the last 30 days?",

#     # Inventory & Operations
#     "How often is each product restocked?",
#     "Which products are consistently low in inventory?",

#     # Customer Journey & Funnels
#     "Average number of page views before a purchase?",
#     "Which products are most often viewed but not purchased?"
# ]

EXAMPLE_QUERIES = [
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?",
    "Average number of page views before a purchase within the last 11 days?",
    "Which products are most often viewed but not purchased in the last 12 days?",
    "How many new users signed up in the past 13 days?",
    "Which campaigns drove the most conversions in the last 14 days?",
    "What regions contributed the most sales in the past 15 days?",
    "Which hours had the highest conversion rate over the last 16 days?",
    "Which users abandoned carts but did not purchase in the last 17 days?",
    "How many refunds were requested in the last 18 days?",
    "Which product categories grew the fastest over the last 19 days?",
    "What is the repeat purchase rate for customers acquired in the last 20 days?",
    "Which traffic sources brought the most new users in the last 21 days?",
    "What is the total number of orders in the last 22 days?",
    "Which products had the highest return rate in the last 23 days?",
    "What are the top 24 most searched keywords in the past 24 days?",
    "Which SKUs experienced inventory shortages in the last 25 days?",
    "What is the average customer lifetime value for users acquired in the last 26 days?",
    "How many support tickets were created in the last 27 days?",
    "Which pages had the highest bounce rate over the past 28 days?",
    "Which products had the highest conversion rate in the last 29 days?",
    "What were the top-performing ads in the last 30 days?",
    "Which product had the most views over the last 1 days?",
    "What was the total revenue for each day over the last 2 days?",
    "How many sessions occurred per hour for the past 3 hours?",
    "Which category has the highest add-to-cart rate in the past 4 days?",
    "How many users placed an order within 5 days of signup?",
    "What is the most popular device type by sales volume in the last 6 days?",
    "Top 7 products by quantity sold this quarter?",
    "Compare average order size by payment method in the last 8 days.",
    "What's the average rating of products sold in the last 9 days?",
    "Which products are consistently low in inventory during the last 10 days?"
]
