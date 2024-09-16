from flask import Flask, request, jsonify
import os
import shutil
import openai
from pdf2image import convert_from_path
from PIL import Image
import base64
import requests
import json

app = Flask(__name__)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def clear_output_dir(output_dir):
    """Clear the output directory if it exists, otherwise create it."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)  # Create the clean output directory


def pdf_to_images(pdf_path):
    """Convert PDF to a list of images (one per page)."""
    images = convert_from_path(pdf_path)
    return images


def save_original_images(images, output_dir):
    """Save the original images from the PDF to the output directory."""
    image_paths = []
    for page_number, image in enumerate(images, start=1):
        output_path = os.path.join(output_dir, f'page_{page_number}.png')
        image.save(output_path)
        image_paths.append(output_path)
    return image_paths


def slice_image(image, coordinates):
    """Slice an image into smaller sections based on the coordinates provided."""
    slices = []
    for coord in coordinates:
        sliced_image = image.crop(coord)
        slices.append(sliced_image)
    return slices


def save_slices(slices, output_dir, page_number):
    """Save the sliced images to the output directory."""
    slice_paths = []
    for i, slice_img in enumerate(slices):
        output_path = os.path.join(
            output_dir, f'page_{page_number}_slice_{i+1}.png')
        slice_img.save(output_path)
        slice_paths.append(output_path)
    return slice_paths


def call_openai_vision_api(image_path, prompt, api_key):
    """Send the sliced image to OpenAI Vision API with a prompt and return the JSON response."""
    openai.api_key = api_key
    base64_image = encode_image(image_path)
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        decoded_response = response.content.decode('utf-8')

        response_json = json.loads(decoded_response)
        assistant_message = response_json['choices'][0]['message']['content']
        return extract_json_from_response(assistant_message)
    except Exception as e:
        return {"error": str(e)}


def extract_json_from_response(assistant_message):
    """Extract valid JSON from the assistant's message."""
    if assistant_message.startswith('```json'):
        json_str = assistant_message[7:].strip()
        if json_str.endswith('```'):
            json_str = json_str[:-3].strip()

    try:
        json_data = json.loads(json_str)
        return json_data
    except json.JSONDecodeError as e:
        return {"error": f"JSON decoding failed: {str(e)}"}


def process_pdf(pdf_path, coordinates, api_key, output_dir='output_slices'):
    """Convert PDF to images, slice each image, and send each slice to OpenAI API."""
    clear_output_dir(output_dir)
    openai.api_key = api_key
    prompt = """Extract the fields from this picture into a clear and valid json.
    • For the checkboxes, use a format "Checkbox_Name": BOOL where BOOL is 1 if checked, 0 if not.
    • if a field is blank, ensure it is recorded as ""
    • only extract existing data,
    do not in any circumstance invent or autocomplete, or correct the existing text or data Start with {"""

    images = pdf_to_images(pdf_path)
    original_image_dir = os.path.join(output_dir, 'original_images')
    os.makedirs(original_image_dir)
    original_image_paths = save_original_images(images, original_image_dir)

    results = []
    for page_number, image in enumerate(images, start=1):
        slices = slice_image(image, coordinates)
        slice_dir = os.path.join(output_dir, f'slices_page_{page_number}')
        os.makedirs(slice_dir)

        slice_paths = save_slices(slices, slice_dir, page_number)
        for slice_path in slice_paths:
            api_response = call_openai_vision_api(slice_path, prompt, api_key)
            results.append({
                'slice_path': slice_path,
                'api_response': api_response
            })

    return results


def combine_results_to_json_string(results):
    """Combine all results into a single JSON string."""
    combined_results = {}

    for result in results:
        for key, value in result['api_response'].items():
            combined_results[key] = value

    json_string = json.dumps(combined_results, indent=4)
    return json_string


@app.route('/process_pdf', methods=['POST'])
def process_pdf_api():
    """API endpoint to process a PDF and return the combined JSON."""
    data = request.json
    pdf_path = data.get('pdf_path')
    # coordinates = data.get('coordinates')
    api_key = data.get('api_key')
    coordinates = [
        (1039, 91, 1573, 299),
        (81, 328, 806, 355),
        (1248, 335, 1601, 356),
        (61, 379, 798, 399),
        (1116, 374, 1598, 399),
        (63, 417, 1161, 440),
        (1184, 424, 1594, 444),
        (65, 461, 837, 477),
        (846, 457, 1593, 472),
        (65, 496, 1592, 506),
        (74, 528, 1415, 577),
        (64, 596, 1594, 612),
        (68, 626, 885, 652),
        (926, 629, 1583, 678),
        (63, 695, 568, 811),
        (624, 695, 808, 809),
        (832, 693, 995, 811),
        (1010, 689, 1204, 815),
        (1217, 693, 1393, 811),
        (1411, 695, 1583, 804)]

    if not pdf_path or not coordinates or not api_key:
        return jsonify({"error": "Missing required parameters."}), 400

    results = process_pdf(pdf_path, coordinates, api_key)
    json_response = combine_results_to_json_string(results)

    return jsonify(json.loads(json_response))


print("Starting Flask app...")
if __name__ == "__main__":
    app.run(debug=True)
