"""Department styling shared by every template that draws a department chip.

The rule cards on two pages were each carrying their own list -- one as a chain
of template conditionals, one as a JavaScript object -- so a department present
in one was a grey box in the other.  Both now read this table, which covers
every department in the catalogue rather than the handful that happened to be
written out.
"""
from __future__ import annotations

import json

from django import template
from django.utils.safestring import mark_safe

register = template.Library()

# emoji, gradient start, gradient end
DEPARTMENT_STYLE = {
    "GROCERY":          ("\U0001F6D2", "#2E86C1", "#3498DB"),
    "DRUG GM":          ("\U0001FA79", "#A569BD", "#BB8FCE"),
    "PRODUCE":          ("\U0001F96C", "#27AE60", "#2ECC71"),
    "COSMETICS":        ("\U0001F484", "#E91E63", "#F06292"),
    "NUTRITION":        ("\U0001F9C3", "#8BC34A", "#AED581"),
    "MEAT":             ("\U0001F969", "#C0392B", "#E74C3C"),
    "MEAT-PCKGD":       ("\U0001F953", "#E74C3C", "#EC7063"),
    "MEAT-WHSE":        ("\U0001F969", "#922B21", "#C0392B"),
    "PORK":             ("\U0001F437", "#C0392B", "#E74C3C"),
    "SEAFOOD":          ("\U0001F41F", "#148F77", "#1ABC9C"),
    "SEAFOOD-PCKGD":    ("\U0001F990", "#0E6655", "#148F77"),
    "DELI":             ("\U0001F96A", "#FF6B6B", "#FF8E8E"),
    "DELI/SNACK BAR":   ("\U0001F32D", "#FF5722", "#FF7043"),
    "DAIRY":            ("\U0001F95B", "#F39C12", "#F8C471"),
    "DAIRY DELI":       ("\U0001F9C0", "#E67E22", "#F0B27A"),
    "BAKERY":           ("\U0001F35E", "#D4AC0D", "#F1C40F"),
    "GRO BAKERY":       ("\U0001F950", "#B7950B", "#D4AC0D"),
    "PASTRY":           ("\U0001F9C1", "#D68910", "#E67E22"),
    "FROZEN GROCERY":   ("\u2744\uFE0F", "#5DADE2", "#AED6F1"),
    "FLORAL":           ("\U0001F490", "#FF69B4", "#FF1493"),
    "GARDEN CENTER":    ("\U0001F331", "#2E7D32", "#66BB6A"),
    "SPIRITS":          ("\U0001F377", "#795548", "#8D6E63"),
    "BEVERAGE":         ("\U0001F964", "#17A2B8", "#20C997"),
    "RESTAURANT":       ("\U0001F37D\uFE0F", "#AD5D2A", "#D68910"),
    "SALAD BAR":        ("\U0001F957", "#4CAF50", "#81C784"),
    "CHEF SHOPPE":      ("\U0001F373", "#FF7043", "#FFAB91"),
    "KIOSK-GAS":        ("\u26FD", "#455A64", "#78909C"),
    "PHARMACY SUPPLY":  ("\U0001F48A", "#8E44AD", "#9B59B6"),
    "RX":               ("\U0001F489", "#7D3C98", "#8E44AD"),
    "HBC":              ("\U0001F9F4", "#9C27B0", "#CE93D8"),
    "HOUSEWARES":       ("\U0001F3E0", "#795548", "#A1887F"),
    "TOYS":             ("\U0001F9F8", "#FF1493", "#FF69B4"),
    "VIDEO":            ("\U0001F4FA", "#9C27B0", "#BA68C8"),
    "VIDEO RENTAL":     ("\U0001F4FC", "#7B1FA2", "#9C27B0"),
    "PHOTO":            ("\U0001F4F7", "#607D8B", "#78909C"),
    "AUTOMOTIVE":       ("\U0001F697", "#37474F", "#607D8B"),
    "ELECT &PLUMBING":  ("\U0001F527", "#FF9800", "#FFB74D"),
    "CNTRL/STORE SUP":  ("\U0001F9F0", "#546E7A", "#90A4AE"),
    "GM MERCH EXP":     ("\U0001F3F7\uFE0F", "#5D6D7E", "#85929E"),
    "POSTAL CENTER":    ("\U0001F4EE", "#1565C0", "#42A5F5"),
    "TRAVEL & LEISUR":  ("\u2708\uFE0F", "#0288D1", "#4FC3F7"),
    "CHARITABLE CONT":  ("\U0001F380", "#C2185B", "#F06292"),
    "COUP/STR & MFG":   ("\U0001F3AB", "#28A745", "#34CE57"),
    "PROD-WHS SALES":   ("\U0001F4E6", "#6D4C41", "#8D6E63"),
    "MISC. TRANS.":     ("\U0001F9FE", "#6C757D", "#868E96"),
    "MISC SALES TRAN":  ("\U0001F6CD\uFE0F", "#6C757D", "#868E96"),
    "UNKNOWN":          ("\u2753", "#6C757D", "#868E96"),
}

DEFAULT_STYLE = ("\U0001F4E6", "#6C757D", "#868E96")


def _style(department):
    return DEPARTMENT_STYLE.get((department or "").strip().upper(), DEFAULT_STYLE)


@register.filter
def dept_icon(department):
    """The emoji standing for a department."""
    return _style(department)[0]


@register.filter
def dept_gradient(department):
    """Its chip background, as a CSS gradient."""
    _, start, end = _style(department)
    return f"linear-gradient(135deg, {start}, {end})"


@register.filter
def dept_color(department):
    return _style(department)[1]


@register.simple_tag
def department_styles_json():
    """The same table for templates that draw their chips in JavaScript."""
    payload = {
        name: {"emoji": emoji, "color": start,
               "gradient": f"linear-gradient(135deg, {start}, {end})"}
        for name, (emoji, start, end) in DEPARTMENT_STYLE.items()
    }
    emoji, start, end = DEFAULT_STYLE
    payload["DEFAULT"] = {"emoji": emoji, "color": start,
                          "gradient": f"linear-gradient(135deg, {start}, {end})"}
    return mark_safe(json.dumps(payload))
