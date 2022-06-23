import json
from inference_onnx import processing_image, predict

def lambda_handler(event, context):
    if 'resource' in event.keys():
        body = event['body']
        body = json.loads(body)
        print(f"Got the input base64")
        image = processing_image(body['b64_image'])
        response = predict(image)
        return {
            'statusCode': 200,
            'headers': {},
            'predicted': json.dumps(response)
        }
    else:
        image = processing_image(event['b64_image'])
        response = predict(image)
        return {
            'statusCode': 200,
            'headers': {},
            'predicted': json.dumps(response)
        }