# -*- coding: utf-8 -*-
"""leggings.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uvHvgPOPrT0yfy1qxm8Kaa5glmPW7GcA
"""

!pip install -q -U google-generativeai

!pip3 install --upgrade google-generativeai

!pip3 install google-cloud-aiplatform

import os
import csv
import google.generativeai as genai
from google.colab import userdata
from PIL import Image
import requests
from io import BytesIO
import requests
import json
import pandas as pd
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import vertexai.preview.generative_models as generative_models
from google.cloud import aiplatform
import numpy as np

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define folder locations
FOLDER_NAME = '/content/drive/My Drive/Leggings/'
# ATTRIBUTES = '/content/drive/My Drive/Leggings/attributes.txt'
DATA_FOLDER = '/content/drive/My Drive/Leggings/leggingsJson'

# Set up the Gemini API
genai.configure(api_key='AIzaSyAb7RfPL0MAe6sJHxKgfW7awgGtvxJdI6w')

# Initialize the Gemini model
generation_config = {
  "temperature": 1.5,
  "top_p": 0.75,
  "top_k": 150,
  "max_output_tokens": 20000,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

def generate_title_template(prompt):
    response = model.generate_content(prompt)
    return response.text.strip()

### USE ONLY ATTRIBUTES AS PLACEHOLDERS TO CREATE TEMPLATE. USE TOP SEARCHED KEYWORDS TO GET THE ORDERING
title_template_prompt = """
Search Terms List (high priority to low): leggings for women, leggings for women ankle length, ankle length leggings for women, jeggings for women, stretchable leggings for women, ankle length ankle length leggings for women, jeggings for women, stretchable tights for women, white leggings for women, leggings go colors, leggings for women combo, shimmer leggings for women, black leggings for women, prisma leggings for women, leggings for women full length, 3/4 leggings for women, jeggings for women, fleece leggings for women, go colors leggings for women, jeggings for women, white leggings for women, leggings leggings for women combo, shimmer leggings for women, black leggings for women, 3/4 leggings for women, prisma leggings for women, leggings for women full length, fleece leggings for women, women leggings, women leggings, leggings for girls, gym leggings for women, leggings for girls, gym leggings for women, ankle leggings for women, go colours leggings for women, ankle length leggings for women, combo pink leggings for women, short leggings for women, cotton leggings for women, black leggings for women, ankle length go colours leggings for women, max leggings for women, ankle leggings for women, white leggings for women, ankle length short leggings for women, ankle length leggings for women, combo pink leggings for women, churidar leggings for women, black leggings for women, ankle length knee length leggings for women, cotton leggings for women, golden leggings for women, lycra leggings for women, white leggings for women, ankle length purple leggings for women, knee length leggings for women, maternity leggings pregnancy, lycra leggings for women, churidar leggings for women, golden leggings for women, printed leggings for women, black leggings for girls, purple leggings for women, maternity leggings pregnancy, printed leggings for women, black leggings for girls, sports leggings for women, leggings for girls 9-10 years, gym leggings for women, high waist leggings for kids, sports leggings for women, leggings for girls 9-10 years, prisma leggings, ankle length leggings, leggings combo, leggings for women, ankle length combo, green leggings for women, blue leggings for women, red leggings for women, white ankle leggings for women, silver leggings for women, avasa leggings for women, shimmer leggings, white leggings, capri leggings for women, leggings combo for women, maroon leggings for women, legging women, leggings ankle length, black ankle leggings for women, black leggings, avaasa leggings for women, prisma ankle leggings for women, grey leggings for women, leggings pants for women, legging for women, ladies leggings, off white leggings for women, half leggings for women, navy blue leggings for women, ankle leggings for women, combo brown leggings for women, sky blue leggings for women, skin colour leggings for women, maternity leggings, 3/4th leggings for women, gold leggings for women, biba leggings for women, baby pink leggings for women, cream leggings for women, leggings for women combo, ankle length lavender leggings for women, layra. leggings go colours, leggings pregnancy, leggings for women, calf length leggings for women, avaasa leggings, go colors leggings for women, ankle length yellow leggings for women, orange leggings for women, leggings ankle length, white ankle length leggings for women, leggings with pockets for women, short leggings for women, knee length go colors leggings, dark green leggings for women, lycra leggings for women, ankle length three fourth leggings, women go colours, ankle length leggings, women leggings combo pack, women's leggings, winter leggings for women, cotton leggings for women combo, leggings for women, 3/4 length beige leggings for women, ankle fit leggings for women, go colors ankle length leggings, combo leggings for women, go colors ankle length leggings for women,
Identify attributes used in Search Terms List and sort the following list (high priority to low) : Brand Name, Product Type, Gender, Bottoms Size, Color Map, Material Type, Rise Style, Fit Type, Special Feature, Seasons, Sport Type, Special Size Type, Leg Style

Donot repeat any attribute or fact more than once

Brand Name is always first.

Provide the response in this EXACT format:
TITLE_TEMPLATE: [attribute] +[attribute] +[attribute] + ....(all given attributes)

"""

# Generate title template
title_template = generate_title_template(title_template_prompt)
print(f"{title_template}")

### USE ONLY FACTS AND ADJECTIVES TO INSERT INTO TEMPLATE. CONNECT USING PREPOSITIONS, PUNCTUATIONS, CONJUCTIONS
prompt = f"""
Facts of the Product : [Leggings],[Girls/Women], [Size: 14/XL], [Shimmer Leggings/Ankle Length Leggings/Cropped Leggings/Denim Leggings/Rib Leggings/Ribbed warm Leggings/Ultra warm Leggings],[All Season]
Adjectives : Sleek, Slimming, Stretchy, Comfortable, Form-fitting, Versatile, Breathable, Supportive, Trendy, High-waisted, Lightweight, Durable, Flexible, Fashionable, Figure-friendly, Sophisticated, Innovative, Trendy, Chic, Essential
Top Searched Keywords : leggings for women, leggings for women ankle length, ankle length leggings for women, jeggings for women, stretchable leggings for women, ankle length ankle length leggings for women, jeggings for women, stretchable tights for women, white leggings for women, leggings go colors, leggings for women combo, shimmer leggings for women, black leggings for women, prisma leggings for women, leggings for women full length, 3/4 leggings for women, jeggings for women, fleece leggings for women, go colors leggings for women, jeggings for women, white leggings for women, leggings leggings for women combo, shimmer leggings for women, black leggings for women, 3/4 leggings for women, prisma leggings for women, leggings for women full length, fleece leggings for women, women leggings, women leggings, leggings for girls, gym leggings for women, leggings for girls, gym leggings for women, ankle leggings for women, go colours leggings for women, ankle length leggings for women, combo pink leggings for women, short leggings for women, cotton leggings for women, black leggings for women, ankle length go colours leggings for women, max leggings for women, ankle leggings for women, white leggings for women, ankle length short leggings for women, ankle length leggings for women, combo pink leggings for women, churidar leggings for women, black leggings for women, ankle length knee length leggings for women, cotton leggings for women, golden leggings for women, lycra leggings for women, white leggings for women, ankle length purple leggings for women, knee length leggings for women, maternity leggings pregnancy, lycra leggings for women, churidar leggings for women, golden leggings for women, printed leggings for women, black leggings for girls, purple leggings for women, maternity leggings pregnancy, printed leggings for women, black leggings for girls, sports leggings for women, leggings for girls 9-10 years, gym leggings for women, high waist leggings for kids, sports leggings for women, leggings for girls 9-10 years, prisma leggings, ankle length leggings, leggings combo, leggings for women, ankle length combo, green leggings for women, blue leggings for women, red leggings for women, white ankle leggings for women, silver leggings for women, avasa leggings for women, shimmer leggings, white leggings, capri leggings for women, leggings combo for women, maroon leggings for women, legging women, leggings ankle length, black ankle leggings for women, black leggings, avaasa leggings for women, prisma ankle leggings for women, grey leggings for women, leggings pants for women, legging for women, ladies leggings, off white leggings for women, half leggings for women, navy blue leggings for women, ankle leggings for women, combo brown leggings for women, sky blue leggings for women, skin colour leggings for women, maternity leggings, 3/4th leggings for women, gold leggings for women, biba leggings for women, baby pink leggings for women, cream leggings for women, leggings for women combo, ankle length lavender leggings for women, layra. leggings go colours, leggings pregnancy, leggings for women, calf length leggings for women, avaasa leggings, go colors leggings for women, ankle length yellow leggings for women, orange leggings for women, leggings ankle length, white ankle length leggings for women, leggings with pockets for women, short leggings for women, knee length go colors leggings, dark green leggings for women, lycra leggings for women, ankle length three fourth leggings, women go colours, ankle length leggings, women leggings combo pack, women's leggings, winter leggings for women, cotton leggings for women combo, leggings for women, 3/4 length beige leggings for women, ankle fit leggings for women, go colors ankle length leggings, combo leggings for women, go colors ankle length leggings for women, layra. leggings ankle length, warm leggings for women, winter comfort leggings for women, shining leggings for women, shiny leggings for women, lace leggings for women, ankle leggings, ankle length leggings for women, prisma cropped leggings for women, black ankle length leggings, women shimmer leggings for women, ankle length gold shimmer leggings for women, branded leggings for women, white leggings for girls, gold colour leggings for women, pink leggings for women, ankle length avasa leggings for women, ankle length leggings, women pocket leggings for women, leggings set for women, fleece leggings, womens leggings, go colours leggings, go colors leggings for women, ankle length go colors leggings, go colours ankle length leggings, go colors ankle length leggings, go colors ankle length leggings for women, prisma leggings, green leggings for women, ankle length leggings, blue leggings for women, leggings combo, white ankle leggings for women, red leggings for women, silver leggings for women, leggings for women, ankle length combo, capri leggings for women, white leggings, leggings pants for women, avasa leggings for women, biba leggings for women, plus size leggings for women, shimmer leggings, leggings combo for women, maroon leggings for women, women leggings, ankle length legging, grey leggings for women, black ankle leggings for women, black leggings, avaasa leggings for women, navy blue leggings for women, prisma ankle leggings for women, off white leggings for women, ladies leggings, legging for women, skin colour leggings for women, half leggings for women, gold leggings for women, flared leggings for women, yellow leggings for women, aurelia leggings for women, ankle leggings for women, combo sky blue leggings for women, 3/4th leggings for women, brown leggings for women, maternity leggings, white ankle length leggings for women, cream leggings for women, comfort leggings for women, calf length leggings for women, layra. leggings orange, leggings for women, baby pink leggings for women, lavender leggings for women, leggings with pockets for women, leggings for women combo, ankle length pregnancy leggings for women, layra. leggings ankle length, three fourth leggings, women avaasa leggings, leggings ankle length, flared leggings, short leggings for women, knee length w leggings for women, lycra leggings for women, ankle length lyra churidar leggings for women, beige leggings for women, max ankle length leggings for women, dark green leggings for women, leggings for women, 3/4 length women's leggings, ankle fit leggings for women, women leggings combo pack, lace leggings for women, winter leggings for women, cotton leggings for women, combo max leggings for women, ankle length combo, leggings for women, cropped leggings for women, shining leggings for women, black ankle length leggings, women warm leggings for women, winter peach leggings for women, shiny leggings for women
Brand Name is : 'GO COLORS'
Based on the summary, images, highly seached keywords, and Title Template Generate the following content in the EXACT format specified:

TITLE: Insert values into {title_template}. Donot Repeat values. Title should be in GOOD ENGLISH by Inserting conjuctions, prepositions, punctuations, adjectives in between. Definitely Insert 'Leggings for Women'.

DESCRIPTION: VERY VERY VERY LONG DESCRIPTION. Write a compelling product description (MUST be 2000 characters)

BULLET_POINT_1: [First key feature or benefit of the product] : VERY VERY LONG BULLET POINT. (600 characterss)

BULLET_POINT_2: [Second key feature or benefit of the product] : VERY VERY LONG BULLET POINT. (600 characterss)

BULLET_POINT_3: [Third key feature or benefit of the product] : VERY VERY LONG BULLET POINT. (600 characterss)

BULLET_POINT_4: [Fourth key feature or benefit of the product] : VERY VERY LONG BULLET POINT. (600 characterss)

BULLET_POINT_5: [Fifth key feature or benefit of the product] : VERY VERY LONG BULLET POINT. (600 characterss)

SEARCH_TERMS: term1, term2, term3, term4, term5, term6, term7, term8, term9, term10 (ONLY 10 comma seperated words)

Please ensure that each section starts with the exact heading (e.g., 'TITLE:', 'DESCRIPTION:', etc.) and is on a new line.
"""

###For rugs : [Brand] + [dimension] + [adjective1]+[pattern]+[colour_map] + [material] +[Carpet/Rug/Dari/Chatai] for+ [room_type1], [room_type2], [room_type3] | [5 FACTS with prepositions, adjectives in between] | [adjective2][special_feature_1] for [occasion1],[occasion2]. (Each word should Start with Capital Letter)

# Function to get image from URL
def get_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# Function to generate content for images and prompt
def generate_content_for_images(prompt, img_urls, summary):
    images = [get_image(url) for url in img_urls]
    full_prompt = f"Summary: {summary} \n\n {prompt}"
    # Assuming model.generate_content() is a placeholder for actual model interaction
    response = model.generate_content([full_prompt] + images)
    print(response.text)  # Assuming model response is printed for debugging
    return response.text.strip()

# Function to parse the generated content
def parse_generated_content(content):
    sections = {
        'TITLE': '',
        'DESCRIPTION': '',
        'BULLET_POINT_1': '',
        'BULLET_POINT_2': '',
        'BULLET_POINT_3': '',
        'BULLET_POINT_4': '',
        'BULLET_POINT_5': '',
        'SEARCH_TERMS': ''
    }

    current_section = None

    for line in content.split('\n'):
        line = line.strip()
        if ':' in line:
            section, content = line.split(':', 1)
            if section in sections:
                current_section = section
                sections[current_section] = content.strip()
        elif current_section:
            sections[current_section] += ' ' + line.strip()

    return sections


def process_json_file(json_file: str, title_template: str) -> List[Dict[str, Any]]:
    """
    Process a single JSON file and generate content for each item.

    Args:
        json_file (str): Name of the JSON file to process.
        title_template (str): Template for generating titles.

    Returns:
        List[Dict[str, Any]]: List of processed results for each item.
    """
    file_results = []
    file_path = os.path.join(DATA_FOLDER, json_file)

    with open(file_path, 'r') as file:
        data = json.load(file)

    for item in data['items']:
        asin = item['asin']
        summary = item['summaries'][0] if item['summaries'] else ''
        images = item['images'][0]['images']

        # Use all images regardless of height and width
        img_urls = [img['link'] for img in images]

        generated_content = generate_content_for_images(prompt, img_urls, summary)
        sections = parse_generated_content(generated_content)

        result = {
            'ASIN': asin,
            'Generated Title': sections['TITLE'],
            'Description': sections['DESCRIPTION'],
            **{f'Bullet Point {i}': sections[f'BULLET_POINT_{i}'] for i in range(1, 6)},
            'Search Terms': sections['SEARCH_TERMS']
        }

        file_results.append(result)

    return file_results

# Get list of JSON files
json_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.json')]

# Process each JSON file sequentially
all_results = []
for json_file in json_files:
    file_results = process_json_file(json_file, title_template)
    all_results.extend(file_results)

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(all_results)
output_csv_path = os.path.join(FOLDER_NAME, 'generated_content.csv')
df.to_csv(output_csv_path, index=False)

print(f"DataFrame saved successfully at: {output_csv_path}")