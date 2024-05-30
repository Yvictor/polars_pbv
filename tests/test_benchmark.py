import polars as pl
import polars_pbv as pl_pbv
import pytest

price_col = [100, 101, 102, 103, 104, 105, 106] * 10000
volume_col = [200, 220, 250, 240, 260, 300, 280] * 10000
df = pl.DataFrame({"price": price_col, "volume": volume_col})
window_size = 120  # 5.0
bins = 20


@pytest.mark.slow
def test_pbv_bench(benchmark):
    @benchmark
    def pbv_bench():
        df.select(
            pl_pbv.pbv_not_par(
                "price", "volume", window_size=window_size, bins=bins, center=False, round=2
            ).alias("pbv"),
        )

@pytest.mark.slow
def test_pbv_par_bench(benchmark):
    @benchmark
    def pbv_par_bench():
        df.select(
            pl_pbv.pbv(
                "price", "volume", window_size=window_size, bins=bins, center=False, round=2
            ).alias("pbv"),
        )