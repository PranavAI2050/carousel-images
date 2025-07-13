from flask import Flask, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os

from utils import (
    collect_valid_images_from_links,
    convert_images_to_llm_strings,
    chunk_descriptions,
    evaluate_chunks_with_llm,
    filter_top_images
)

app = Flask(__name__)

@app.route('/generate_content', methods=['POST'])
def generate_content():
    try:
        print("[INFO] Received request at /generate_content")
        data = request.get_json()
        links = data.get('links_to_be_search', [])

        # Setup headless browser
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=options)
        print("[INFO] Headless Chrome driver initialized")

        # Prepare fake results_list format if needed by util
        results_list = [{'title': 'User Link', 'link': link} for link in links]

        # Collect images
        final_images = collect_valid_images_from_links(links, results_list, driver)
        print(f"[INFO] Collected {len(final_images)} images")

        # Convert to LLM-friendly strings and get image URLs
        desc_strings, img_links = convert_images_to_llm_strings(final_images)

        # Run LLM-based filtering
        chunks = chunk_descriptions(desc_strings)
        evaluated = evaluate_chunks_with_llm(chunks, topic=None)  # topic optional
        top_images = filter_top_images(evaluated, img_links)
        print(f"[INFO] Filtered down to {len(top_images)} top images")

        return jsonify({
            "top_images": top_images,
            "image_sources": img_links
        })

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        driver.quit()
        print("[INFO] Driver closed")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4000))
    app.run(host='0.0.0.0', port=port)
