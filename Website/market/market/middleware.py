import re

SCRIPT_BLOCK = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.IGNORECASE | re.DOTALL)
STYLE_BLOCK = re.compile(r"(<style\b[^>]*>)(.*?)(</style\s*>)", re.IGNORECASE | re.DOTALL)
HTML_COMMENT = re.compile(r"<!--(?!\[if\b).*?-->", re.DOTALL)
CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
PLACEHOLDER = "\x00script-{}\x00"


def strip_css_comments(text):
    return CSS_COMMENT.sub("", text)


def strip_html_comments(html):
    scripts = []

    def hold(match):
        scripts.append(match.group(0))
        return PLACEHOLDER.format(len(scripts) - 1)

    guarded = SCRIPT_BLOCK.sub(hold, html)
    guarded = HTML_COMMENT.sub("", guarded)
    guarded = STYLE_BLOCK.sub(
        lambda m: m.group(1) + strip_css_comments(m.group(2)) + m.group(3), guarded
    )

    for index, script in enumerate(scripts):
        guarded = guarded.replace(PLACEHOLDER.format(index), script, 1)
    return guarded


class StripCommentsMiddleware:
    

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        if getattr(response, "streaming", False) or not hasattr(response, "content"):
            return response

        content_type = response.get("Content-Type", "")
        if content_type.startswith("text/html"):
            cleaner = strip_html_comments
        elif content_type.startswith("text/css"):
            cleaner = strip_css_comments
        else:
            return response

        charset = response.charset or "utf-8"
        try:
            body = response.content.decode(charset)
        except UnicodeDecodeError:
            return response

        response.content = cleaner(body).encode(charset)
        if response.has_header("Content-Length"):
            response["Content-Length"] = str(len(response.content))
        return response
