import argparse
from sagemaker.sklearn import SKLearn
from getconfig import CLOUD_CONFIG


def main(args):
    print("args.local=", args.local)
    # Initialise SDK
    sklearn_estimator = SKLearn(
        entry_point='src/train_and_deploy.py',
        role=CLOUD_CONFIG['sagemaker_role_id']['value'],
        train_instance_type='local' if args.local else 'ml.m4.xlarge',
        hyperparameters={
            'sagemaker_submit_directory': f"s3://{CLOUD_CONFIG['s3bucket']['value']}",
        },
        framework_version='0.23-1',
        metric_definitions=[
            {'Name': 'train:score', 'Regex': 'train:score=(\S+)'}],
    )
    # Run model training job
    sklearn_estimator.fit({
        'train': "file://./data/data.csv" if args.local else f"s3://{CLOUD_CONFIG['s3bucket']['value']}/data.csv"
    })

    # Deploy trained model to an endpoint
    sklearn_estimator.deploy(
        instance_type='local' if args.local else 'ml.t2.medium',
        initial_instance_count=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local',
                        action='store_true',
                        help="deploy in local environment")
    args = parser.parse_args()
    main(args)
