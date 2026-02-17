import json
import os

try:
    with open('feed_output.json', 'r', encoding='utf-16') as f:
        data = json.load(f)
    
    images = []
    if 'data' in data:
        for item in data['data']:
            if item.get('image'):
                images.append(item['image'])
    
    with open('image_list.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(images))
    print(f"Extracted {len(images)} image URLs.")
except Exception as e:
    print(f"Error: {e}")
