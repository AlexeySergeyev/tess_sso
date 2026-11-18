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

- Walkthrough of the whole pipeline on one observation (asteroid 95,
  sector 71 by default; set the asteroid and observation in its first
  cell): [examples/demo.ipynb](examples/demo.ipynb), script version
  [examples/demo.py](examples/demo.py).
- Validation on three observations, against the reference pipeline:
  [examples/validation.ipynb](examples/validation.ipynb).
- Injection and recovery of a known signal (criteria 3 and 5):
  [examples/injection.ipynb](examples/injection.ipynb).
- From pixels to a binary-asteroid search: binaries' whole analysis of the
  asteroids downloaded here with tess_sso as its photometry
  (`photometry: tess_sso`): [examples/binaries.ipynb](examples/binaries.ipynb),
  script version [examples/binaries.py](examples/binaries.py).
- 4337 Arecibo (faint), binaries' hand-tuned analysis on the reference and
  tess_sso light curves, residuals compared:
  [examples/asteroid_4337.ipynb](examples/asteroid_4337.ipynb), script
  version [examples/asteroid_4337.py](examples/asteroid_4337.py).
- 3485 Barucci, a template from the name to the rotation period: the TESS
  visits, the download, the photometry, the period from all observations
  together, and the light curves (set the asteroid in its first cell):
  [examples/asteroid_3485.ipynb](examples/asteroid_3485.ipynb), script
  version [examples/asteroid_3485.py](examples/asteroid_3485.py).
- Pipeline, and how the background and the stars are removed: [docs/pipeline.md](docs/pipeline.md).
- The same steps on real data (3486, sector 45), with images of each model:
  [examples/background_and_stars.ipynb](examples/background_and_stars.ipynb).
- Design: [docs/design-decisions.md](docs/design-decisions.md).
- Roadmap and phase plans: [docs/plans/](docs/plans/2026-09-24-implementation-roadmap.md).

Tests: `.venv/bin/python -m pytest` (offline),
`.venv/bin/python -m pytest -m network` (JPL, MAST, MIT, Gaia), and
`.venv/bin/python -m pytest -m validation` (the example notebooks on whole
observations).

binaries (package `tessbin`) uses tess_sso when a target or observation
sets `photometry: tess_sso` in `data/targets/<asteroid>.yml`; see binaries'
`docs/api/extraction.md`.
