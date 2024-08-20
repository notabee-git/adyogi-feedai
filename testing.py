from flask import Flask, request, jsonify
import pandas as pd
import google.generativeai as genai
from PIL import Image
import requests
from io import BytesIO
import re
import emoji
import logging
from google.cloud import bigquery
import traceback

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up the Gemini API
try:
    genai.configure(api_key='YOUR_GEMINI_API_KEY')
    logger.info("Gemini API configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {str(e)}")
    raise

# Initialize the Gemini model
generation_config = {
    "temperature": 1.5,
    "top_p": 0.75,
    "top_k": 150,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

try:
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    logger.info("Gemini model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini model: {str(e)}")
    raise

# Initialize BigQuery client
try:
    bigquery_client = bigquery.Client()
    logger.info("BigQuery client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize BigQuery client: {str(e)}")
    raise

def fetch_product_data(client_id, category):
    query = f"""
    SELECT asin, values, img_links
    FROM `your_project.your_dataset.your_table`
    WHERE client_id = '{client_id}' AND category = '{category}'
    """
    try:
        query_job = bigquery_client.query(query)
        results = query_job.result()
        
        df = results.to_dataframe()
        logger.info(f"Successfully fetched {len(product_data)} products for client {client_id} and category {category}")
        return df
    except Exception as e:
        logger.error(f"Error fetching product data: {str(e)}")
        raise

# Function to filter images based on dimensions and extract links
def extract_links(data):
    links = []
    if isinstance(data, list):
        for item in data:
            if 'images' in item:
                links.extend([img['link'] for img in item['images'] if img['height'] >= 500 and img['width'] >= 500])
    return links

# Function to extract specific values
def extract_values(data):
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        item = data[0]
        return [{
            'autographed': item.get('autographed'),
            'brand': item.get('brand'),
            'color': item.get('color'),
            'itemName': item.get('itemName'),
            'size': item.get('size'),
            'style': item.get('style'),
            'displayName': item.get('browseClassification', {}).get('displayName')
        }]
    return {}

def clean_format(cell):
    if isinstance(cell, list) and len(cell) > 0 and isinstance(cell[0], dict):
        return '\n'.join([f"{key.capitalize()}: {value}" for key, value in cell[0].items()])
    return cell

def load_images(urls):
    images = []
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            images.append(img)
        except requests.RequestException as e:
            logger.error(f"Error loading image from {url}: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing image from {url}: {str(e)}")
    return images

def analyze_images(images, content):
    try:
        response = model.generate_content([content] + images)
        return response.text
    except Exception as e:
        logger.error(f"Error analyzing images: {str(e)}")
        raise

def content(category, values, search_terms, product_template):
    return f"""
    Your role: You are a helpful Amazon SEO expert designed to assist the user in creating high performance product listings for {category} in amazon marketplace.
    I will be presenting you a list of product attributes, their corresponding values, search terms and image/images for each product.
    Task: Your task is to verify if the below listed values accurately reflect the information depicted in the image/images. If there is any discrepancy, you should provide the correct values.
    Output the values after correcting
    Values:
    {values}

    Search Terms:
    {search_terms}

    IMPORTANT: Provide a clean and structured response without any special characters except ':'

    Now, based on the following product response and attributes, generate a flashy and attention-grabbing title (up to 250 characters), a rich and captivating detailed description (up to 2000 characters), detailed bullet points (up to 500 characters each), and search terms (up to 200 characters).
    Mention below are the rules to be followed strictly while generating the title, description, bulle points and search terms.
    RULE 1: The brand name should come first in the title.
    RULE 2: The template of the title is - {product_template}
    RULE 3: Do include punctuations, prepositions, pronouns and conjunctions in the title.
    RULE 4: While generating the title, description, bullet points and search terms, utilise the full character length limit. Ensure the character length does not exceed the specified limit.

                  Title:
                  Description:
                  Bullet Points:
                  (Higlighted Bullet Point title): Bullet Point Details
                  (Higlighted Bullet Point title): Bullet Point Details
                  (Higlighted Bullet Point title): Bullet Point Details
                  (Higlighted Bullet Point title): Bullet Point Details
                  (Higlighted Bullet Point title): Bullet Point Details
                  Search Terms:
    """

def process_product(product_data, category, search_terms, product_template):
    product_name = product_data['asin']
    product_values = product_data['values']
    product_urls = product_data['img_links']

    max_retries = 3
    attempts = 0
    success = False

    while attempts < max_retries and not success:
        try:
            images = load_images(product_urls)
            logger.info(f"Analyzing {product_name}...")
            content_text = content(category, product_values, search_terms, product_template)
            result = analyze_images(images, content_text)
            success = True
            logger.info(f"Successfully analyzed {product_name}")
            return result
        except Exception as e:
            attempts += 1
            logger.error(f"Error analyzing {product_name}: {str(e)}. Retrying {attempts}/{max_retries}...")

    logger.error(f"Failed to analyze {product_name} after {max_retries} attempts.")
    return None

def post_process_result(result):
    try:
        result = re.sub(r"[#\*\*]", "", result).strip()
        result = emoji.demojize(result).strip()
        return result
    except Exception as e:
        logger.error(f"Error post-processing result: {str(e)}")
        raise

def extract_content(cell, start_tag, end_tag):
    try:
        start_index = 0
        end_index = len(cell)

        if f"{start_tag}:" in cell:
            start_index = cell.index(f"{start_tag}:") + len(f"{start_tag}:")
        elif f"{start_tag}" in cell:
            start_index = cell.index(f"{start_tag}") + len(f"{start_tag}")

        if f"{end_tag}:" in cell:
            end_index = cell.index(f"{end_tag}:", start_index)
        elif f"{end_tag}" in cell:
            end_index = cell.index(f"{end_tag}", start_index)

        return cell[start_index:end_index].strip()
    except ValueError as e:
        logger.warning(f"Error extracting content for tags {start_tag} and {end_tag}: {str(e)}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error extracting content: {str(e)}")
        raise

@app.route('/analyze_products', methods=['POST'])
def analyze_products():
    try:
        data = request.json
        client_id = data.get('client_id')
        category = data.get('category')
        product_template = data.get('product_template')
        search_terms = data.get('search_terms')

        if not all([client_id, category, product_template, search_terms]):
            logger.warning("Missing required parameters in request")
            return jsonify({"error": "Missing required parameters"}), 400

        logger.info(f"Analyzing products for client {client_id}, category {category}")

        # Fetch product data from BigQuery
        product_df = fetch_product_data(client_id, category)

        if not product_df:
            logger.warning(f"No products found for client {client_id} and category {category}")
            return jsonify({"error": "No products found for the given client_id and category"}), 404

        results = []

        # create links, values, clean and store relevant data
        product_df['img_links'] = product_df['images'].apply(lambda x: extract_links(x) if isinstance(x, list) else [])
        product_df['values'] = product_df['summaries'].apply(extract_values)
        product_df['amazon_links'] = product_df['asin'].apply(lambda x: f"https://www.amazon.com/dp/{x}")
        product_df['values'] = product_df['values'].apply(clean_format)
        product_df = product_df[['asin','amazon_links', 'img_links', 'values']]

        for product_data in product_df:
            try:
                result = process_product(product_data, category, search_terms, product_template)
                
                if result is not None:
                    processed_result = post_process_result(result)
                    analysis = {
                        'asin': product_data['asin'],
                        'title': extract_content(processed_result, "Title", "Description"),
                        'description': extract_content(processed_result, "Description", "Bullet Points"),
                        'bullet_points': extract_content(processed_result, "Bullet Points", "Search Terms"),
                        'search_terms': extract_content(processed_result, "Search Terms", ".")
                    }
                    results.append(analysis)
                else:
                    results.append({
                        'asin': product_data['asin'],
                        'error': "Failed to analyze product"
                    })
            except Exception as e:
                logger.error(f"Error processing product {product_data['asin']}: {str(e)}")
                results.append({
                    'asin': product_data['asin'],
                    'error': f"Error processing product: {str(e)}"
                })

        logger.info(f"Successfully analyzed {len(results)} products")
        return jsonify(results)

    except Exception as e:
        logger.error(f"Unexpected error in analyze_products: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)