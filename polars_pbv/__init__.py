from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polars_pbv.utils import parse_into_expr, register_plugin, parse_version

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

if parse_version(pl.__version__) < parse_version("0.20.16"):
    from polars.utils.udfs import _get_shared_lib_location

    lib: str | Path = _get_shared_lib_location(__file__)
else:
    lib = Path(__file__).parent


def pbv_not_par(
    price: IntoExpr,
    volume: IntoExpr,
    window_size: int,
    bins: int,
    center: bool = True,
    round: int = -1,
) -> pl.Expr:
    price = parse_into_expr(price)
    volume = parse_into_expr(volume)
    return register_plugin(
        args=[price, volume],
        symbol="pbv_not_par",
        is_elementwise=False,
        lib=lib,
        kwargs={
            "window_size": window_size,
            "bins": bins,
            "center_label": center,
            "round": round,
        },
    )

def pbv(
    price: IntoExpr,
    volume: IntoExpr,
    window_size: int,
    bins: int,
    center: bool = True,
    round: int = -1,
) -> pl.Expr:
    price = parse_into_expr(price)
    volume = parse_into_expr(volume)
    return register_plugin(
        args=[price, volume],
        symbol="pbv",
        is_elementwise=False,
        lib=lib,
        kwargs={
            "window_size": window_size,
            "bins": bins,
            "center_label": center,
            "round": round,
        },
    )

def pbv_pct(
    price: IntoExpr,
    volume: IntoExpr,
    window_size: int,
    bins: int,
    center: bool = True,
    round: int = -1,
) -> pl.Expr:
    price = parse_into_expr(price)
    volume = parse_into_expr(volume)
    return register_plugin(
        args=[price, volume],
        symbol="pbv_pct",
        is_elementwise=False,
        lib=lib,
        kwargs={
            "window_size": window_size,
            "bins": bins,
            "center_label": center,
            "round": round,
        },
    )


def pbv_topn_vp(
    price: IntoExpr,
    volume: IntoExpr,
    window_size: int,
    bins: int,
    n: int,
    center: bool = True,
    round: int = -1,
) -> pl.Expr:
    price = parse_into_expr(price)
    volume = parse_into_expr(volume)
    return register_plugin(
        args=[price, volume],
        symbol="pbv_topn_vp",
        is_elementwise=False,
        lib=lib,
        kwargs={
            "window_size": window_size,
            "bins": bins,
            "n": n,
            "center_label": center,
            "round": round,
            "pct": False,
        },
    )


def pbv_topn_v(
    price: IntoExpr,
    volume: IntoExpr,
    window_size: int,
    bins: int,
    n: int,
    center: bool = True,
    round: int = -1,
    pct: bool = False,
) -> pl.Expr:
    price = parse_into_expr(price)
    volume = parse_into_expr(volume)
    return register_plugin(
        args=[price, volume],
        symbol="pbv_topn_v",
        is_elementwise=False,
        lib=lib,
        kwargs={
            "window_size": window_size,
            "bins": bins,
            "n": n,
            "center_label": center,
            "round": round,
            "pct": pct,
        },
    )
