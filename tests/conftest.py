import os

import certifi

# Point requests/urllib at certifi's bundle for any tests that do hit the network.
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
