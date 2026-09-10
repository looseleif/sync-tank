# Public aquarium website

The public project page is `index.html` at the repository root. Open it directly
in a browser to preview it. Its aquarium-workshop layout uses the original
colorful fish artwork, a filterable build gallery, and a small pixel aquarium.
Serif headings, ordinary underlined links and restrained borders give it an
older personal-site feel without sale signs, novelty badges or mock shop copy.
It uses plain HTML, CSS, JavaScript, and five existing project images. The catalog
links to documentation and experiments; no products are sold here. It does not
connect to tank nodes, show live feeds, or control hardware.

Keep the colorful original artwork as the banner. Gallery images should document
a specific project activity: habitat setup, bench testing, hardware assembly,
interface experiments, or installation. Captions must distinguish earlier
prototypes from the current software and tests from verified results. The
software catalog photograph shows the actual dry-bench test, not the browser
builder or a live feed from this website.

`hardware.html` explains the electronics, product references, provenance gaps and
connections. `builder.html` provides the photo-reference 3D tank sandbox with
local draft storage and JSON import/export. See [the modeling guide](../docs/TANK_MODELING.md).

The builder uses JavaScript modules, so preview it over HTTP rather than opening
it as a local file. From the repository root:

```bash
python -m http.server 8876 --bind 127.0.0.1
```

Open `http://127.0.0.1:8876/builder.html`. Photo files stay in the browser session;
only geometry and reference metadata are saved in the draft.

## GitHub Pages

In the repository's **Settings > Pages > Build and deployment**, select
**GitHub Actions** as the source. Run **Publish aquarium website** from the
Actions tab once after enabling Pages. Subsequent changes to the page, its
assets, or its workflow deploy automatically from `main`.

Expected URL: https://looseleif.github.io/sync-tank/

The workflow publishes the three public HTML pages, their styles and scripts,
the six referenced images, and the existing Three.js/OrbitControls modules with
third-party notices. Python services, deployment configuration, archives, and
tank data are not part of the Pages artifact. Hosting the live hub still requires
a PC or Raspberry Pi on the local network.

## Verify changes

The Pages workflow checks local asset links and JavaScript syntax before upload.
Run `python site/tests/check_assets.py` from the repository root for the same link
check. For browser checks, with the preview server running:

```bash
python -m pip install playwright pillow
python -m playwright install chromium
python site/tests/check_builder.py http://127.0.0.1:8876
```

Use `--channel chrome` to test an installed Chrome instead. The browser check
covers unplaced inventory, movement/aim, face changes, photo cropping, JSON
validation, local draft recovery, canvas pixels, and desktop/mobile layouts.
It runs without real tank nodes or motors.
