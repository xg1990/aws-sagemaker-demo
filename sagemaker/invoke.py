import boto3

# https://aws.amazon.com/blogs/machine-learning/call-an-amazon-sagemaker-model-endpoint-using-amazon-api-gateway-and-aws-lambda/

def main():
    client = boto3.client('sagemaker-runtime')
    response = client.invoke_endpoint(
        EndpointName='<EndpointName>',
        Body=b"[[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,53,6,7,8,9,10]]",
        ContentType='application/json',
        Accept='application/json',
    )
    print("response['Body']=", response['Body'].read())
if __name__ == "__main__":
    main()