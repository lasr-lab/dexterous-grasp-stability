# Project Page

This directory contains the source for the [project page](https://lasr-lab.github.io/dexterous-grasp-stability/) of the paper *Temporal Visuo-Tactile Learning for Dexterous Grasp Stability* (arXiv preprint coming soon).

The design and implementation can be reused for other research projects under the [template license](LICENSE).

## Design

The page uses a single continuous layout: text, figures, and videos appear together, with no section collapsed or hidden behind tabs. Plain HTML, CSS, and JavaScript keep it small and easy to edit, with no framework or build step.

GitHub Pages deploys from either a branch's root or its `/docs` folder, and `/docs` keeps the repository root free for research code, such as `src/`, scripts, and configuration. Everything the site serves lives in this directory, so no separate branch or build workflow is needed, and the page can be published first with code added later. Research and presentation stay together, each with its own space.

The same arrangement holds for larger codebases maintained alongside their website. A separate repository makes sense when maintainers or release schedules differ, and a dedicated branch when site changes need their own history.

## Editing

- Content, links, and BibTeX: [index.html](index.html)
- Styles: [css/style.css](css/style.css)
- Behavior: [js/main.js](js/main.js)
- Figures: [images/](images/)
- Clips: [videos/](videos/)

`.nojekyll` turns off the Jekyll build, which would otherwise treat `{{` in the BibTeX entry as template syntax.

## Publishing

In the repository on GitHub, select **Settings → Pages → Deploy from a branch → main → /docs**; see the [GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) for details.

## License

The template (the HTML, CSS, and JavaScript here) is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); see [LICENSE](LICENSE). The research content shown on the page (figures, clips, and the prose describing the work) is not covered.

The icons in the publication links and the BibTeX copy button are from [Feather v4.29.2](https://github.com/feathericons/feather/tree/v4.29.2) and are separately licensed under MIT. They are embedded inline, with stroke width adjusted to match the page. See [NOTICE](NOTICE) for the source details and full license text.

This template was independently written, not derived from an existing one. The layout follows a structure now widely adopted by academic project pages such as [Nerfies](https://nerfies.github.io/).
