# AR-VLA Website

Static HTML/CSS website for showcasing the AR-VLA paper, core idea, results, and experiment videos.

## Local Preview

Open `index.html` directly in a browser, or run a simple static server.

Example with Python:

```bash
python3 -m http.server 8080
```

Then visit `http://localhost:8080`.

## Cloudflare Pages Deployment

Use the following settings in Cloudflare Pages:

- Framework preset: `None`
- Build command: *(leave empty)*
- Build output directory: `/`

This repository is already deployable as a static site without a build step.

## Button Links

Current placeholder links for `Paper`, `Code`, and `Checkpoints` all point to:

- https://arxiv.org/abs/2603.10126

Update these links directly in `index.html` when final URLs are ready.
