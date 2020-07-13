# AWS SageMaker Example

This is the source code for the Medium article
https://medium.com/weareservian/machine-learning-on-aws-sagemaker-53e1a5e218d9

Before successfully runing this code, you may need to fill in your personal configurations (e.g. S3 bucket name)

## Pre-requisite

* Python3
* AWS SDK
* Docker
* Terraform (optional)

## Setup AWS resources

``` bash
cd terraform
terraform init # for the first time
terraform apply
terraform output --json > ../sagemaker/cloud_config.json
cd ..
```

## Generate Demo data

``` bash
python data/generate.py --output data/data.csv
```

## Test in local environment

``` bash
python sagemaker/jobsubmit.py --local
```

### Invocate Local endpoint

``` bash
curl --location --request POST '127.0.0.1:8080/invocations' \
     --header 'Content-Type: application/json' \
     --data-raw '[[1,2,3,4,5,6,7,8,19,10],[1,2,3,4,5,6,7,8,9,10]]'
```

### Upload data to s3 bucket

``` bash
aws s3 cp data/data.csv s3://$(cd terraform && terraform output s3bucket)/
```

## Submit to AWS cloud

``` bash
python sagemaker/jobsubmit.py
```

### Invocate Remote endpoint

``` bash
python sagemaker/invoke.py
```

## Clean up Cloud environment

``` bash
cd terraform
terraform destroy
cd ..
```
