# customers/templatetags/custom_filters.py
from django import template

register = template.Library()

@register.filter
def format_time(value):
    """Converts a decimal like 1631.00 to a string '16:31'."""
    if value is None:
        return ""
    time_str = str(int(value)).zfill(4)
    return f"{time_str[:2]}:{time_str[2:]}"

@register.filter(name='get_item')
def get_item(dictionary, key):
    """
    Allows getting a dictionary item by key in the template.
    Usage: {{ my_dictionary|get_item:my_key }}
    """
    if hasattr(dictionary, 'get'):
        return dictionary.get(key)
    return None