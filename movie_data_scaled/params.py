"""
Purpose:    Read adn format environment variables
Import:     from .params import env
Execute:    python3 -m <package> "$@"
Use:        env.VAR_NAME
"""

import os


class Environment:
    # AWS
    AWS_ACCESS_KEY = str(os.getenv('AWS_ACCESS_KEY', ''))
    AWS_SECRET_KEY = str(os.getenv('AWS_SECRET_KEY', ''))
    REGION = str(os.getenv('us-west-1', ''))
