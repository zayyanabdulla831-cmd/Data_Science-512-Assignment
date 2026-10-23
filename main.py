import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def scrape_books(max_pages=15):

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    scraped_data = []

    print("Starting web scraping...")
    for page in range(1, max_pages + 1):
        url = f"https://books.toscrape.com/catalogue/page-{page}.html"

        try:
            response = requests.get(url, headers=headers)
            response.encoding = 'utf-8'  # Prevent character encoding issues

            if response.status_code != 200:
                print(f"Page {page} returned status code {response.status_code}. Stopping pagination.")
                break

            soup = BeautifulSoup(response.text, 'html.parser')
            books = soup.find_all('article', class_='product_pod')

            if not books:
                break

            for book in books:
                title = book.find('h3').find('a')['title']
                raw_price = book.find('p', class_='price_color').text

                rating_elem = book.find('p', class_='star-rating')
                rating_word = rating_elem['class'][1] if rating_elem and len(rating_elem['class']) > 1 else 'Zero'

                stock_elem = book.find('p', class_='in stock availability')
                stock_status = stock_elem.text.strip() if stock_elem else 'Unknown'

                scraped_data.append({
                    'Title': title,
                    'Raw_Price': raw_price,
                    'Rating_Word': rating_word,
                    'Stock_Status': stock_status
                })
            print(f"Successfully scraped page {page} ({len(books)} books).")

        except Exception as e:
            print(f"Error scraping page {page}: {e}")

    df = pd.DataFrame(scraped_data)
    print(f"Total books scraped: {len(df)}")
    return df

df_raw = scrape_books(max_pages=15)

def clean_catalogue_data(df):

    if df.empty:
        raise ValueError("Scraping returned no data. Check your internet connection or URL structure.")

    df['Price_GBP'] = (df['Raw_Price']
                       .astype(str)
                       .str.replace('£', '', regex=False)
                       .str.replace('Â£', '', regex=False)
                       .str.replace('$', '', regex=False)
                       .str.strip()
                       .astype(float))

    rating_mapping = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    df['Rating'] = df['Rating_Word'].map(rating_mapping).fillna(0).astype(int)

    df['In_Stock'] = df['Stock_Status'].apply(lambda x: 1 if 'In stock' in str(x) else 0)

    df['Title_Length'] = df['Title'].astype(str).apply(len)

    cleaned_df = df[['Title', 'Price_GBP', 'Rating', 'In_Stock', 'Title_Length']].dropna()
    return cleaned_df

df_clean = clean_catalogue_data(df_raw)
print("\nCleaned Dataset Sample:")
print(df_clean.head())

def generate_visualizations(df):

    sns.set_theme(style="white grid")

    plt.figure(figsize=(8, 5))
    sns.histplot(df['Price_GBP'], bins=20, kde=True, color='teal')
    plt.title('Distribution of Book Prices (£)')
    plt.xlabel('Price in GBP (£)')
    plt.ylabel('Frequency')
    plt.savefig('price_distribution.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Rating', y='Price_GBP', data=df, palette='Set2')
    plt.title('Book Price across Star Ratings')
    plt.xlabel('Star Rating (1-5)')
    plt.ylabel('Price in GBP (£)')
    plt.savefig('price_vs_rating.png')
    plt.close()


generate_visualizations(df_clean)
print("\nVisualizations successfully generated and saved as PNG files.")

def train_pricing_model(df):

    x = df[['Rating', 'In_Stock', 'Title_Length']]
    y = df['Price_GBP']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation Results ---")
    print(f"Model Coefficients: {model.coef_}")
    print(f"Model Intercept: {model.intercept_:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R^2) Score: {r2:.4f}")

    return model


pricing_model = train_pricing_model(df_clean)