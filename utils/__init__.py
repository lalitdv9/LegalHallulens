"""
LegalHalluLens utilities.

Helpers for the typed hallucination auditing and calibrated multi-agent debate
pipeline described in the LegalHalluLens paper (AIWILD @ ICML 2026, arXiv:2606.18021).
"""

__version__ = "0.1.0"
__author__ = "Lalit Yadav, Akshaj Gurugubelli"

# Configure SSL certificates globally to fix certificate verification errors
import os
import ssl

try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
except ImportError:
    # If certifi is not installed, fall back to system certificates
    pass

