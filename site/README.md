# Public aquarium website

The public project page is `index.html` at the repository root. Open it directly
in a browser to preview it. It uses plain HTML, CSS, JavaScript, and five existing
project images. It does not connect to tank nodes, show live feeds, or control
hardware. The small pixel aquarium is a simulation.

## GitHub Pages

In the repository's **Settings > Pages > Build and deployment**, select
**GitHub Actions** as the source. Run **Publish aquarium website** from the
Actions tab once after enabling Pages. Subsequent changes to the page, its
assets, or its workflow deploy automatically from `main`.

Expected URL: https://looseleif.github.io/sync-tank/

The workflow publishes only the HTML, stylesheet, aquarium script, and the five
referenced images. Python services, deployment configuration, archives, and tank
data are not part of the Pages artifact. Hosting the live hub still requires a
PC or Raspberry Pi on the local network.
