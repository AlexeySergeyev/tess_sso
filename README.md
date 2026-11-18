# tess_sso

Photometry of asteroids in TESS full-frame images: find the observations,
download only the pixels along the track, model the scattered light and the
stars, and measure the asteroid with a joint PRF fit per group of exposures.

```python
from tess_sso.asteroid import Asteroid
from tess_sso.settings import Settings

ast = Asteroid("95", settings=Settings(group_minutes=30))
ast.observations()                      # TESS visits; nothing downloaded
ast.plan(71, 2, 2).describe()           # pixels and GB still to download
ast.download(71, 2, 2)
phot = ast.photometry(71, 2, 2)         # models, then flux per 30-minute group
phot.to_table()
ast.lightcurve(71, 2, 2, photometry=phot)   # tess-95-s0071-2-2-lc.fits, for lightkurve and binaries
ast.pixel_product(71, 2, 2, photometry=phot)  # tess-95-s0071-2-2-tpf.fits: group stacks, models, residuals
```

Data go to `<project>/data/tess_sso/<asteroid>/` (override with
`TESS_SSO_DATA`); deleting an asteroid's folder frees everything downloaded
for it. Files already on disk are used without asking MAST again; network
steps time out and are retried. On a network whose IPv6 resolves but does
not connect, set `TESS_SSO_PREFER_IPV4=1` (or call
`tess_sso.network.prefer_ipv4()`) so requests do not wait for IPv6 first.
HTTPS certificates are checked against the system's trust store (through
`truststore`), not OpenSSL's `/etc/ssl/cert.pem`, which on macOS lacks the
roots of MAST and NAIF; set `TESS_SSO_SYSTEM_CERTS=0` to keep Python's
default. A failed certificate check is raised at once, not retried.

Tests: `python -m pytest` (offline),
`python -m pytest -m network` (JPL, MAST, MIT, Gaia), and
`python -m pytest -m validation` (whole observations compared with the
reference pipeline; needs its reference files, path in `BINARIES_DATA`).

binaries (package `tessbin`) uses tess_sso when a target or observation
sets `photometry: tess_sso` in `data/targets/<asteroid>.yml`; see binaries'
`docs/api/extraction.md`.
