"""tess_sso: photometry of asteroids in TESS full-frame images."""

import os as _os

__version__ = "0.1.0"

if _os.environ.get("TESS_SSO_SYSTEM_CERTS", "") != "0":
    from tess_sso.network import use_system_certificates as _use_system_certificates

    _use_system_certificates()

if _os.environ.get("TESS_SSO_PREFER_IPV4", "") not in ("", "0"):
    from tess_sso.network import prefer_ipv4 as _prefer_ipv4

    _prefer_ipv4()
