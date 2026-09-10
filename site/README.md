# Public aquarium website

The public project page is `index.html` at the repository root. Open it directly
in a browser to preview it. Its retro aquarium-shop layout uses the original
colorful fish artwork, a filterable project catalog, and a small pixel aquarium.
It uses plain HTML, CSS, JavaScript, and five existing project images. The catalog
links to documentation and experiments; no products are sold here. It does not
connect to tank nodes, show live feeds, or control hardware.

## GitHub Pages

In the repository's **Settings > Pages > Build and deployment**, select
**GitHub Actions** as the source. Run **Publish aquarium website** from the
Actions tab once after enabling Pages. Subsequent changes to the page, its
assets, or its workflow deploy automatically from `main`.

Expected URL: https://looseleif.github.io/sync-tank/

The workflow publishes only the HTML, stylesheet, aquarium and catalog scripts, and the five
referenced images. Python services, deployment configuration, archives, and tank
data are not part of the Pages artifact. Hosting the live hub still requires a
PC or Raspberry Pi on the local network.
