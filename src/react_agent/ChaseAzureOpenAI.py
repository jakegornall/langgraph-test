from langchain_openai import AzureChatOpenAI
import os
from datetime import datetime, timedelta, timezone
import base64
import binascii
import jwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography import x509
import requests
from dotenv import load_dotenv
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

cert_path = os.path.realpath('./OctagonAI-ServicePrincipal.pem')
root_ca_path = os.path.realpath('./JPMCROOTCA.pem')

# Proxies
JPMC_PROXY = {
    'http': 'proxy.jpmchase.net:10443',
    'https': 'proxy.jpmchase.net:10443',
}

def get_cert_details(cert_data, password=None):
    private_key = serialization.load_pem_private_key(
        cert_data, password, default_backend())

    cert = x509.load_pem_x509_certificate(cert_data, default_backend())
    sha1 = binascii.hexlify(cert.fingerprint(hashes.SHA1())).decode()
    x5t = base64.b64encode(bytearray.fromhex(sha1))

    return {
        "private_key": private_key,
        "cert_thumbprint": x5t.decode()
    }


def encode_jwt(tenant_id, client_id, cert_thumbprint, private_key):
    encoded_jwt = jwt.encode({
        "aud": f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
        "iss": client_id,
        "sub": client_id,
        "jti": datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S%f'),
        "nbf": datetime.now(tz=timezone.utc),
        "exp": datetime.now(tz=timezone.utc) + timedelta(seconds=30)
    },
        private_key,
        "RS256",
        {
            "x5t": cert_thumbprint,
            "alg": "RS256",
            "typ": "JWT"
        })
    return encoded_jwt

def generate_jwt(tenant_id, client_id, cert_data, password=None):
    cert_details = get_cert_details(cert_data, password)
    jwt_token = encode_jwt(tenant_id, client_id,
                           cert_details["cert_thumbprint"],
                           cert_details["private_key"]
                           )
    return jwt_token

def auth_request(url, payload, verify): # added "verify"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = requests.request(
        "GET", url,
        headers=headers,
        data=payload,
        proxies=JPMC_PROXY,
        verify=verify # added JPMC root CA
    )

    access_token = response.json()["access_token"]
    return access_token

def using_cert(tenant_id, client_id, jwt_token, scope, verify): # added "verify"
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

    payload = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'scope': scope,
        'client_assertion_type': 'urn:ietf:params:oauth:client-assertion-type:jwt-bearer',
        'client_assertion': jwt_token
    }
    return auth_request(url, payload, verify) # added "verify"

def get_access_token():
    scope = "https://cognitiveservices.azure.com/.default"
    tenant_id = os.environ["AZURE_TENANT_ID"]
    client_id = os.environ["AZURE_SPN_CLIENT_ID"]

    # to fetch cert from secret manager
    # cert = get_spn_key()

    # to fetch cert from local file
    cert = open(cert_path, "rb").read()

    jwt_token = generate_jwt(tenant_id, client_id, cert, None)

    # Path to the root CA .pem file
    access_token = using_cert(tenant_id, client_id, jwt_token, scope, root_ca_path)
    
    return access_token

def get_access_token_headers():
    access_token = get_access_token()

    return {
        "Authorization": f"Bearer {access_token}"
    }

def headers():
    return

def get_openai_parameters(model_name):
    access_token = get_access_token()
    common_params = {
        "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "openai_api_version": os.environ["AZURE_OPENAI_API_VERSION"],
        "deployment_name": os.environ["AZURE_OPENAI_MODEL"],
        "openai_api_key": os.environ["AZURE_OPENAI_API_KEY"],
        "openai_api_type": "azure",
        "default_headers": { "Authorization": f"Bearer {access_token}", "user_sid": "REPLACE" }
    }

    if any(substring in model_name for substring in ["o1-mini", "o3-mini"]):
        specific_params = {
            "temperature": 1,
            "max_completion_tokens": 100000
        }
    elif any(substring in model_name for substring in ["gpt-4o"]):
        specific_params = {
            "temperature": 0,
            "max_tokens": 16384
        }
    else:
        print("Unknown model:", model_name, "; using default parameters.")
        specific_params = {
            "temperature": 0,
            "max_tokens": 16384
        }

    return {**common_params, **specific_params}

def getModel():
    model_name = os.environ["AZURE_OPENAI_MODEL"]
    print("Loading LLM model:", os.environ["AZURE_OPENAI_MODEL"])
    model_params = get_openai_parameters(model_name)
    print(f"Loaded optimized model params: Temperature: {model_params['temperature']}, Max tokens: {model_params['max_tokens'] if 'max_tokens' in model_params else model_params['max_completion_tokens']}")

    return AzureChatOpenAI(**model_params)

if __name__ == "__main__":
    get_access_token()
    model = getModel()

    # get base64 data from ./test.png
    # with open("./src/react_agent/test.png", "rb") as image_file:
    #   image_data = image_file.read()
    #   base64_data = base64.b64encode(image_data).decode('utf-8')
    # 
    # # save base64 to a txt file
    # with open("./src/react_agent/base64.txt", "w") as text_file:
    #   text_file.write(base64_data)

    result = model.invoke([
        HumanMessage(content=[
            {"type": "text", "text": "What is the React JavaScript framework?"},

            # {
            #    "type": "image_url",
            #    "image_url": {
            #        "url": f"data:image/png;base64,{base64_data}",
            #        "detail": "high"
            #    }
            # }
        ]),
    ])

    print(result.content)
