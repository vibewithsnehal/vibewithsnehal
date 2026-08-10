"""Build the single-file app: inline core.js into index.html -> dist/courtvision-app.html."""

from pathlib import Path

here = Path(__file__).parent
html = (here / "index.html").read_text()
core = (here / "core.js").read_text()
html = html.replace('<script src="core.js"></script>', f"<script>\n{core}\n</script>")
dist = here / "dist"
dist.mkdir(exist_ok=True)
out = dist / "courtvision-app.html"
out.write_text(html)
print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
