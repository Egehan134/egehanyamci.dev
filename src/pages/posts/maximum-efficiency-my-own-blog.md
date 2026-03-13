---
layout: ../../layouts/BlogPostLayout.astro
title: "Maximum Efficiency: My Own Blog"
date: "2026-03-13"
tags: []
toc:
  - id: "introduction-why-my-own-blog"
    text: "Introduction"
  - id: "technology-choice-astro"
    text: "Technology Choice: Astro"
  - id: "reasons-why-i-chose-astro"
    text: "Reasons Why I Chose Astro"
    children:
      - id: "zero-javascript-by-default"
        text: "Zero JavaScript By Default"
      - id: "efficiency-oriented-approach"
        text: "Efficiency-Oriented Approach"
      - id: "content-oriented-structure"
        text: "Content-Oriented Structure"
      - id: "island-architecture"
        text: "Island Architecture"
  - id: "islands-on-my-site"
    text: "'Islands' on My Site"
  - id: "mini-astro-guide"
    text: "Mini Astro Guide"
  - id: "site-architecture"
    text: "Site Architecture"
    children:
      - id: "basic-layout-structure"
        text: "Basic Layout Structure"
      - id: "sidebar-component"
        text: "Sidebar Component"
      - id: "component-based-architecture"
        text: "Component-Based Architecture"
      - id: "d-table-of-contents-toc-and-three-column-layout"
        text: "TOC and Three Column Layout"
  - id: "theme-system"
    text: "Theme System"
  - id: "comment-system"
    text: "Comment System"
  - id: "data-management"
    text: "Data Management"
  - id: "deployment"
    text: "Deployment"
  - id: "conclusion"
    text: "Conclusion"


      
---

Maximum Efficiency: My Own Blog

## Introduction: Why My Own Blog?

First of all, I need to explain why I decided to start blogging. I have experienced that writing something readable, fluent, and truly meaningful is very difficult. Preparing articles, like many other things, is not something that can be done without an internal desire.

I have always loved showing and explaining what I do. However, after a certain point, explaining it to the people around me was no longer enough. Therefore, the idea of sharing my work in the form of a blog began to seem increasingly meaningful.

At the same time, the idea of being able to read what I wrote again years later is something I like. In a way, this blog serves as an archive for me. When I look back in the future, I will be able to see what I was doing and how I was thinking during that period.

Another effect I didn't expect was discipline. The desire to keep this site active pushes one to work. Finding something to write requires producing something. I realized that while this pushes one to work and produce, it also provides discipline.

So, while there are ready-made platforms (Medium, Dev.to, etc.), why did I decide to build my own site from scratch?

The answer is quite simple: I wanted a playground that was completely under my control. I wanted to create an area where I could touch every point from its design to its architecture. At the same time, this site became a small “playground” where I could reflect my technical skills.

Even though HTML or frontend development are not normally the areas that interest me the most, the idea of making a visually pleasing and useful site was quite attractive.

## Technology Choice: Astro

To be honest, this blog is a fairly simple system. It does not host a large-scale content platform or a complex backend architecture. Therefore, even if Next.js, Hugo, or even plain HTML/CSS were used, the site would likely run fast enough.

However, my goal was not just to create a **"fast enough"** site. I wanted to build the **most efficient and lightweight architecture** possible.

To me, it felt technically unnecessary for the browser to load a large JavaScript framework (React, Vue, etc.) in the background just to read a blog post. Since most of the content is already static, this approach seemed too costly.

This is exactly where **Astro** became a very logical choice for me.

####  Why I Didn't Settle for "Fast Enough"?

Most modern web frameworks handle pages with an SPA (Single Page Application) approach. Even for a static page, the browser:
1. Downloads the HTML
2. Loads JavaScript bundles
3. Starts the hydration process
During hydration, the HTML content is "revived" and made interactive by JavaScript. 
This approach is quite powerful, but it's not always necessary for content-heavy sites. Running hundreds of kilobytes of JavaScript on the user's device just to read a blog post seemed architecturally too costly to me.

## Reasons Why I Chose Astro

### A) Zero JavaScript by Default

My favorite feature of Astro is that it does not send JavaScript to the browser by default. Written components are converted to static HTML during the build phase. Thus, the lightest possible page is sent to the user.
Thanks to this approach:
- Faster first load,
- Lower JavaScript cost,
- Better Core Web Vitals,<br>
can be achieved.

### B) Efficiency-Oriented Approach

Even if the site is small, I didn't want to load unnecessary resources for the user. With Astro, only what is truly necessary is sent to the user:
* HTML
* CSS
* small pieces of JavaScript if needed<br>
This approach provides a significant advantage, especially in the site's performance scores.

### C) Content-Oriented Structure

One of the most important reasons I chose Astro is that it offers a development experience very suitable for content production. It natively supports:
* Markdown
* MDX
* File-based routing<br>
In this way, instead of dealing with complex routing configurations or database operations, it is possible to add posts just by creating a new .md file.

### E)   Island Architecture

One of Astro's most powerful features is the Island Architecture approach. In this architecture, most of the page is generated as static HTML. Only small components requiring interaction are run with JavaScript.

We can think of it like this:
Page → static HTML "sea"<br>
Interactive components → JavaScript "islands"

In this way, performance is maintained while interactive features can be added when necessary.

## "Islands" on My Site

JavaScript is used on my site only where it is truly necessary.

**Theme Toggle**
The theme switch button must be functional as soon as the page loads. Therefore, I configured this component to run client-side.

**Mobile Menu**
The mobile navigation menu also runs client-side as it depends on user interaction.

**Giscus Comment System**
Since the comment system is at the very bottom of the page, it is unnecessary to download it during the initial load. Therefore, the comment system is loaded using IntersectionObserver only when the user reaches the bottom of the page. This approach helps maintain the initial page load performance.

## Mini Astro Guide

Apart from the topics I wrote about here, I specifically wanted to mention this feature of Astro.
Astro is not just another JavaScript framework; it is a "static-first" meta-framework that adopts a performance-centric approach to modern web.

### Component Anatomy: Code Fence Structure

An .astro file, unlike traditional components, is divided into two distinct sections.

#### Server-Side Script (Frontmatter)

The section between the --- blocks at the top of the component runs only during the build stage (or on the server if SSR is used).
```astro
---
// This section is not sent to the browser!
import MyComponent from './MyComponent.astro';
const response = await fetch('[https://api.data.com/posts](https://api.data.com/posts)');
const data = await response.json();
---
```

**Why Is It Important?** Database queries, API calls, or secret key (API Key) usage performed here never reach the end-user's browser. This is a massive advantage for both security and performance.

#### Component Template

The section below the frontmatter uses a JSX-like syntax, but the result is pure HTML.
```astro
<div>
  <h1>{data.title}</h1>
  <MyComponent />
</div>
```
## Site Architecture: Sidebar, Layout, and Three-Column Structure

I designed the general architecture of the site to be quite simple yet sustainable. Making the architecture simple, understandable, and sustainable is something that should be done in every project, not just for this site. Even years later, I should be able to look back and understand what I did without overthinking.

### A) Basic Layout Structure

In this structure developed using Astro, I created a common Layout component so that all pages offer a consistent user experience. Pages are rendered through this central layout.

#### Simplified Hierarchy of the Architecture:
```text
|- Layout
    |- Sidebar (Navigation and Controls)
    |- Main Content (Main content area)
        |- Page Content (Page-Specific Content)
```

To put it more simply, the Layout stands at the base; we layer the Sidebar and the Main Content over it. Then, we layer the Page Content over the Main Content. We can think of these as parts that overlap each other. Each layered part follows the rules of the part beneath it.

With this approach, when we create a new page, it is enough to focus only on the content; the Sidebar, theme system, and other global components are automatically included in the structure. This minimizes code repetition and provides great convenience by preventing us from rebuilding every page.

### B) Sidebar Component

At the center of the site structure, there is a multi-functional Sidebar component located on the left side of the screen. This component is not just a navigation section, but also a part of the site's fundamental layout integrity.

#### Sidebar Content:
- Main Navigation Links
- Contact Links
- Theme switch button

#### Sticky Navigation and User Experience
To increase usability, I configured the Sidebar to remain fixed (position: sticky) when the page is scrolled. I wanted the navigation to be particularly easy to access.

```astro
.sidebar {
  position: sticky;
  top: 0;
}
```

#### Responsive Behavior
While the Sidebar stands as a fixed column on the left side in the desktop view, I decided to hide it and move it into a hamburger menu on mobile devices to use the screen area efficiently. I arranged it so that the menu opens when the button is pressed and automatically closes when the user clicks on a link.

### C) Component-Based Architecture

The sidebar and layout structure are in a completely modular architecture where the interface is divided into small and manageable parts:

- components/Sidebar.astro
- components/ThemeToggle.astro
- components/Comments.astro<br><br>

Components are like reusable Lego pieces that make up a website. Instead of building the entire page as a single massive block, we design the button, menu, or a product card as separate "building blocks." In this way, when we want to use the same button in ten different places on the site, we don't write the code ten times; we just call that part. This keeps the code cleaner and makes the project much easier to manage.

Additionally, the biggest advantage of this architecture is that it makes maintenance easier thanks to the "Separation of Concerns" principle. For example, when we make an improvement in the theme-switching mechanism, according to the "Single Responsibility" principle, it is enough to only update the ThemeToggle component; this change does not affect other parts.

### D) Table of Contents (TOC) and Three-Column Layout

To facilitate in-page navigation within the articles, I added a Table of Contents (TOC) component. With the addition of the TOC, the initial 2-column layout (Sidebar | Content) evolved into a 3-column structure.

#### Why Three Columns?
Initially, I placed the TOC component inside the content area. However, doing so led to problems like different appearances based on screen width, constant changes in content width, and the breaking of sticky behavior.

Layout Schema:
```text
+-----------+-------------------+-----------+ 
|  Sidebar  |      Content      |    TOC    | 
| (Global)  |    (Blog Post)    | (Headings)| 
+-----------+-------------------+-----------+
```

### Technical Implementation
I used CSS Grid to establish the three-column structure and ensure responsive flexibility. The "minmax(0, 1fr)" expression used for the middle column allows the content to grow flexibly while preventing overflows.

```astro
.layout {
  display: grid;
  grid-template-columns: 240px minmax(0, 1fr) 220px;
  gap: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.sidebar, .toc {
  position: sticky;
  top: 2rem;
  height: fit-content;
}
```

### Gradual Responsive Transition
As the screen width narrows, the layout adapts as follows:

**Wide (Desktop):** 3 columns (Sidebar | Content | TOC)<br>
**Slightly Narrow (Tablet):** 2 Columns (TOC is hidden, only Sidebar and Content remain)<br>
**Narrow (Mobile):** 1 Column (Sidebar is moved to the menu, only Content is shown)<br>

```astro
@media (max-width: 1024px) {
  .layout {
    grid-template-columns: 220px 1fr;
  }
  .toc {
    display: none;
  }
}
```

This structure keeps both global navigation and in-page heading tracking accessible for every screen size

## Theme System

I wanted to go beyond a standard “Light/Dark Mode” option in the site's user interface and offer different visual atmospheres. For this, I added various themes that I personally like and are quite popular. In addition to the normal Light/Dark mode, I added Tokyo Night (Cloud), Gruvbox (Box), and Forest (Leaf) themes, and they looked much better on the site than I expected.

#### Technical Implementation and Data Persistence:
I made the theme system sustainable within itself, just like the rest of the site.

**CSS Variables:** All of the site's color palettes are managed through central CSS variables. When the theme changes, only the values of these variables are updated.

**Data-Theme Attribute:** The selected theme is added to the HTML <html> tag as a data-theme attribute. CSS selectors activate different color sets based on this attribute.

**LocalStorage Integration:** I store the user's selected theme preference in the browser's localStorage. In this way, even if the page is refreshed or the user visits the site again later, the chosen atmosphere is preserved.

This structure allows the theme switching process to occur instantly while making it quite simple to add new themes on the code side.

## Comment System

I believe a comment section for interaction was essential. I didn't want to write a comment system from scratch, handle storage, etc. I decided to use [Giscus](https://giscus.app/), which is open-source, easy to use and integrate, and uses GitHub Discussions as a backend. Using Giscus is truly simple; you can find its documentation on its website.

Giscus was great in every way, but there was a minor issue for my site. Since Giscus runs inside an iframe (an HTML tag used to embed another web page or document inside a web page), a technical difficulty arose in syncing it with the site's overall theme changes.

The problem was that the iframe is isolated. When the site theme changes, the CSS variables in the main document cannot directly affect the third-party iframe where Giscus runs. This caused the comment area to remain in the old theme even when the site theme changed.

To solve this problem and ensure it works flawlessly, I set up a two-stage mechanism:

1. **MutationObserver:** Using MutationObserver on the JavaScript side, the data-theme attribute of the html tag is constantly monitored. A trigger function runs whenever a change is detected.

2. **postMessage API:** The detected theme change is transmitted to the Giscus iframe as a secure message via the postMessage method. The Giscus component, receiving the message, instantly synchronizes its internal style with the site's new theme (e.g., dark_dimmed or light).

Even though the fact that the comments section's color always stays the same might not even be noticed by most users, I wanted it to be perfect. Thanks to this solution, the moment a user changes the theme, the comment area immediately adapts to the site's new visual atmosphere without requiring a page refresh.

## Data Management

For the sustainability of the site, I adopted the principle of "Separation of Concerns" between content and code. This structure both facilitates data management and accelerates the content production process.

### Project Portfolio:

Instead of being written directly into HTML pages, the projects featured in the project cards on the site are maintained as a centralized data structure (Array of Objects) in the src/data/projects.js file.

This way, adding a new project or updating an existing one requires no coding or design work; knowing how to code is not even necessary. Simply updating the data file is enough. The projects added here are rendered dynamically; the UI reads this data file and automatically generates the project cards.

### Blog Content:

I store blog posts in the developer-friendly Markdown (.md) format. Every new .md file I add to the src/pages/posts directory is automatically scanned using the Astro.glob() API. This function extracts the frontmatter (title, date, description, etc.) data from the files and dynamically generates the blog post list for both the homepage and the blog section. Each added .md file is automatically processed by Astro and rendered as an independent page with its own URL structure.

While I automated these parts, one more tedious task remained: adding headings to the TOC. To simplify this, I created a separate structure (which still needs some improvement). Here is how it works:

1. Astro's Markdown processor automatically extracts headings such as h2 and h3 within the post and provides them as an array.

2. These extracted headings are passed as props to the TOC component within the Layout structure. The component uses this data to dynamically create in-page navigation links.

3. While writing, there is no need to manually create a "Table of Contents" table or link headers by hand. Most of the time (though the current version sometimes requires manual adjustments), simply creating headings according to Markdown standards is enough for the system to build the entire navigation.

These convenience features were very important to me. Since writing posts and working on projects already take up a lot of time, I didn't want to spend additional time writing code just to add them to the blog. Building a system that requires zero code was what satisfied me the most.

## Deployment

Delivering the content to the internet is just as important as the site's architecture and content itself. Instead of dealing with inefficient processes like manually uploading files or managing servers, I established a CI/CD (Continuous Integration / Continuous Deployment) structure that complies with modern software development standards.

The operation of this system is quite simple and effective:

- GitHub Integration: All source codes and blog posts (Markdown files) of the site are stored in a GitHub repository.

- Automatic Triggering: The process begins the moment I write a post or make a code change on my local computer and push it to GitHub.

- Cloudflare Pages: Fully integrated with GitHub, Cloudflare Pages immediately detects this change. It builds the Astro project on its own servers (converting it into static files) and goes live globally within seconds.

Thanks to this structure, instead of wasting time with infrastructure management, I can focus entirely on producing content and developing new features. Updating the entire site in seconds with a single command perfectly complements the project's "zero-code management" philosophy. Furthermore, by using GitHub and Cloudflare infrastructure, I avoid server costs. The only cost for maintaining the site is the domain fee, which I pay a small annual amount for.

## Conclusion

This blog is not just a platform for me to share my writings. It has also become a place where I can track my technical growth. Taking notes while working on a project and turning that project into an article is much more enjoyable than it appears. Seeing the knowledge I have accumulated here is an incredible source of motivation.

Technically speaking, this work has become a model showing how a modern website should be created from both technical and user experience perspectives. I believe the layout and component-based structure I developed are truly professional. Designing in a modular way has always been a satisfying design approach for me. I think the ability to change everything without damaging the overall system is a significant feature.

Over time, I plan to add a few more features to the site. Currently, I need an in-site search system and a newsletter system. I will add those one day. You can, of course, find the code for this site on [GitHub](https://github.com/Egehan134/egehanyamci.dev). I am sure my code is much more understandable than my writing!