import os
import ssl

# Set TensorFlow logging level to minimize verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Suppress XLA logs
os.environ["XLA_FLAGS"] = "--xla_hlo_profile=false"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

# Disable SSL verification (not recommended for production)
if os.getenv("ENV", "dev") == "dev":
    ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
