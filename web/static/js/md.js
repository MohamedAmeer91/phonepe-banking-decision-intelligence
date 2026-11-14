// Tiny markdown subset renderer (paragraphs, bullets, tables, bold, code). Escapes all HTML first.
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const inline = (s) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

export function renderMarkdown(text) {
  const lines = (text || "").split("\n");
  let html = "", i = 0, depth = 0;
  const closeLists = (to = 0) => { while (depth > to) { html += "</ul>"; depth--; } };
  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      closeLists();
      const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const head = cells(line); i += 2;
      html += '<div class="tablewrap"><table><thead><tr>' + head.map((c) => "<th>" + inline(c) + "</th>").join("") + "</tr></thead><tbody>";
      while (i < lines.length && /^\s*\|/.test(lines[i])) {
        html += "<tr>" + cells(lines[i]).map((c) => "<td>" + inline(c) + "</td>").join("") + "</tr>"; i++;
      }
      html += "</tbody></table></div>";
      continue;
    }
    const m = line.match(/^(\s*)- (.*)$/);
    if (m) {
      const want = m[1].length >= 2 ? 2 : 1;
      while (depth < want) { html += "<ul>"; depth++; }
      closeLists(want);
      html += "<li>" + inline(m[2]) + "</li>";
    } else if (line.trim() === "") {
      closeLists();
    } else {
      closeLists();
      html += "<p>" + inline(line.trim()) + "</p>";
    }
    i++;
  }
  closeLists();
  return html;
}
