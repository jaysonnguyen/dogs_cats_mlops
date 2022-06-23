import json
from inference_onnx import processing_image, predict

def lambda_handler(event, context):
    b64_value = event['b64_image']
    print(f"Got the input base64")
    image = processing_image(b64_value)
    response = predict(image)
    return {
        'statusCode': 200,
        'headers': {},
        'predicted': response
    }
