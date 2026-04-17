# [egehanyamci.dev](https://egehanyamci.dev/) | Blog & Portfolio

This repository contains the source code for my personal technical blog, built with a focus on minimalism, performance, and technical clarity. This site hosts technical documentation, project case studies, and insights focused on software engineering, data science, and other technologies.

## Tech Stack

The project leverages a static site generation workflow to ensure optimal performance and maintainability:

- **Framework:** [Astro](https://astro.build/) (Static Site Generation)
- **Deployment:** [Cloudflare Pages](https://pages.cloudflare.com/)
- **Styling:** Tailwind CSS
- **Content:** Markdown & MDX

## Project Structure

```text
├── public/          # Static assets (images, icons, etc.)
├── src/
│   ├── components/  # Reusable UI components
│   ├── layouts/     # Page templates
│   ├── data/        # Markdown blog posts and technical notes
│   ├── styles/      # Global css and theme config
│   └── pages/       # Routes and views
│        └── posts   # Blog posts
└── astro.config.mjs # Astro configuration
```

#### For a detailed technical breakdown of how this blog was engineered for maximum efficiency, you can read the dedicated post:

* [Maximum Efficiency: My Own Blog](https://egehanyamci.dev/posts/maximum-efficiency-my-own-blog/) - An analysis of the architectural decisions and implementation details in this project.