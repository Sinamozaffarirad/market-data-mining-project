from pathlib import Path
path = Path("templates/site/dunnhumby/basket_analysis.html")
text = path.read_text(encoding="utf-8")
old = "function showRuleInsertFeedback(message, variant) {\n  const container = document.getElementById('descriptiveInsertAlert');\n  if (!container) {\n    alert(message);\n    return;\n  }\n  container.textContent = message;\n  container.className = `alert alert-${variant}`;\n  container.classList.remove('d-none');\n  setTimeout(() => {\n    container.classList.add('d-none');\n  }, 5000);\n}\n"
if old not in text:
    raise SystemExit('Original feedback function not found')
new = "function showRuleInsertFeedback(message, variant) {\n  const container = document.getElementById('descriptiveInsertAlert');\n  if (!container) {\n    console.log(`[${variant}] ${message}`);\n    return;\n  }\n  container.textContent = message;\n  container.className = `alert alert-${variant}`;\n  container.classList.remove('d-none');\n  setTimeout(() => {\n    container.classList.add('d-none');\n  }, 5000);\n}\n"
path.write_text(text.replace(old, new), encoding="utf-8")
