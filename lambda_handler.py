import json
from inference_onnx import base64_to_image, processing_image, predict

def lambda_handler(event, context):
    body = event['body']
    body = json.loads(body)
    b64_value = body['b64_image']
    print(f"Got the input base64: {b64_value}")
    image_path = base64_to_image(b64_value)
    image = processing_image(image_path)
    response = predict(image)
    return {
        'statusCode': 200,
        'headers': {},
        'predicted': response
    }
