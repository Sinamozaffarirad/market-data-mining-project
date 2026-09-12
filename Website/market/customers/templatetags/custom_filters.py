# customers/templatetags/custom_filters.py
from django import template

register = template.Library()

@register.filter
def format_time(value):

    if value is None:
        return ""
    time_str = str(int(value)).zfill(4)
    return f"{time_str[:2]}:{time_str[2:]}"

@register.filter(name='get_item')
def get_item(dictionary, key):
    
    if hasattr(dictionary, 'get'):
        return dictionary.get(key)
    return None