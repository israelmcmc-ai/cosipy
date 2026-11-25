# boto3 client for COSI's public Wasabi bucket, with the read-only credentials cosipy's fetch_wasabi_file uses.
import inspect
import boto3
from cosipy.util import data_fetching

_defaults = {k: v.default for k, v in inspect.signature(data_fetching.fetch_wasabi_file).parameters.items()}
B = _defaults['bucket']


def client():
    return boto3.client('s3', endpoint_url=_defaults['endpoint'],
                        aws_access_key_id=_defaults['access_key'], aws_secret_access_key=_defaults['secret_key'])
