# nba_model_generator

### Serverless commands to create a service with python 3.8 template in the folder ./myService
serverless create --template aws-python3 --path myService

### Deploy service to AWS Lambda
sls deploy
serverless deploy

### Setting up environment 
npm -g install serverless
Create IAM user
sls config credentials --provider aws --key {{Access key ID}} --secret {{Secret access key}}

