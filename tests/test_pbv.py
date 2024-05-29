import polars as pl
from polars_pbv import pbv

def test_pbv():
    price_col = [100, 101, 102, 103, 104, 105, 106, ]#107, 108, 109]
    volume_col = [200, 220, 250, 240, 260, 300, 280, ]# 270, 310, 330]
    window_size = 6#5.0
    bins = 3
    df = pl.DataFrame({
        'price': price_col,
        'volume': volume_col
    })
    expected_df = pl.DataFrame({
        "price": [*[None,]*5, [100., 101.666667, 103.333333], [101., 102.666667, 104.333333]],
        "volume": [*[None,]*5, [200+220, 250+240, 260+300], [220+250, 240+260, 280+300]]
    }).select(
        pl.struct("price", "volume").alias("pbv")
    )
    result = df.select(
        pbv("price", "volume", window_size=window_size, bins=bins, center=False).alias("pbv"),
        # pl.col("price").pbv.pbv(volume=pl.col("volume"), window_size=window_size, bins=bins).alias("pbv")
    )
    print(df)
    assert result.equals(expected_df) 