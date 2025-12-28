import json
import boto3
import urllib.parse

s3 = boto3.client('s3')
textract = boto3.client('textract')

def lambda_handler(event, context):
    # Get the uploaded object info from the event
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'])

    print(f"Triggered by file: s3://{bucket_name}/{key}")

    try:
        # Call Textract to extract text
        response = textract.detect_document_text(
            Document={'S3Object': {'Bucket': bucket_name, 'Name': key}}
        )

        # Extract text lines
        extracted_text = ""
        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                extracted_text += block['Text'] + '\n'

        # Create a new file name
        output_key = f"extracted-{key.rsplit('.', 1)[0]}.txt"

        # Upload extracted text to S3
        s3.put_object(
            Bucket=bucket_name,
            Key=output_key,
            Body=extracted_text.encode('utf-8')
        )

        print(f"Extracted text uploaded to s3://{bucket_name}/{output_key}")
        return {
            'statusCode': 200,
            'body': json.dumps('Text extraction and upload successful.')
        }

    except Exception as e:
        print(e)
        raise e
