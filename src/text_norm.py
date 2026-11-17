"""Normalize only the configured input text; never append labels or metadata."""
import re
import polars as pl


def normalize_text(value, cfg, intermediary=None):
    text = re.sub(cfg.d['build']['whitespace_pattern'], ' ', (value or '').lower()).strip()
    if cfg.d['build']['add_intermediary'] and intermediary:
        text = re.sub(cfg.d['build']['whitespace_pattern'], ' ', f'{text} {intermediary.lower()}').strip()
    return text


def text_expr(cfg):
    expr = pl.col('key_strict')
    if cfg.d['build']['add_intermediary']:
        expr = pl.concat_str([expr, pl.col('intermediary').fill_null('')], separator=' ')
    return expr.str.to_lowercase().str.replace_all(cfg.d['build']['whitespace_pattern'], ' ').str.strip_chars().alias('text')
