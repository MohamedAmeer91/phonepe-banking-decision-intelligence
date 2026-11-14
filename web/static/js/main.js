import * as pages from "./pages.js";

const routes = { "": pages.overview, part1: pages.part1, part2: pages.part2, part3: pages.part3, part4: pages.part4, memo: pages.memo, limitations: pages.limitations };

async function render() {
  const key = location.hash.replace(/^#\/?/, "");
  const app = document.getElementById("app");
  const page = routes[key];
  document.querySelectorAll("#nav a").forEach((a) => {
    if (a.dataset.route === key) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  if (!page) { app.innerHTML = '<h1>Page not found</h1><p><a href="#/">Back to overview</a></p>'; return; }
  app.innerHTML = '<p class="muted">Loading...</p>';
  try {
    await page(app);
    window.scrollTo(0, 0);
  } catch (e) {
    app.innerHTML = '<div class="error" role="alert">Could not load this page: ' + String(e.message).replace(/</g, "&lt;") + "</div>";
  }
}
window.addEventListener("hashchange", render);
render();
