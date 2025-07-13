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

@app.route('/get_images', methods=['POST'])
def get_images():
    try:
        print("[INFO] Received request at /get_images")
        data = request.get_json()
        links = data.get('links_to_be_search', [])
        topic = data.get('topic', '')  # <-- Now accepting topic from input

        print(f"[DEBUG] Topic: {topic}")
        print(f"[DEBUG] Links received: {len(links)}")

        # Setup headless browser
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=options)
        print("[INFO] Headless Chrome driver initialized")

        # Prepare results_list (if required by your util)
        results_list = [{'title': 'User Link', 'link': link} for link in links]

        # Collect images
        final_images = collect_valid_images_from_links(links, results_list, driver)
        print(f"[INFO] Collected {len(final_images)} images")

        # Convert image objects into descriptions and links
        desc_strings, img_links = convert_images_to_llm_strings(final_images)

        # Run LLM-based filtering
        chunks = chunk_descriptions(desc_strings)
        evaluated = evaluate_chunks_with_llm(chunks, topic=topic)  # <-- Topic used here
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
