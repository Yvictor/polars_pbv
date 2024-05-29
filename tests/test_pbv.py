import polars as pl
from polars_pbv import pbv, pbv_pct, pbv_topn_vp, pbv_topn_v


def test_pbv():
    price_col = [100, 101, 102, 103, 104, 105, 106]
    volume_col = [200, 220, 250, 240, 260, 300, 280]
    window_size = 6  # 5.0
    bins = 3
    df = pl.DataFrame({"price": price_col, "volume": volume_col})
    expected_df = pl.DataFrame(
        {
            "price": [
                *[
                    None,
                ]
                * 5,
                [100.0, 101.67, 103.33],
                [101.0, 102.67, 104.33],
            ],
            "volume": [
                *[
                    None,
                ]
                * 5,
                [200 + 220, 250 + 240, 260 + 300],
                [220 + 250, 240 + 260, 280 + 300],
            ],
        }
    ).select(pl.struct("price", "volume").alias("pbv"))
    result = df.select(
        pbv(
            "price", "volume", window_size=window_size, bins=bins, center=False, round=2
        ).alias("pbv"),
        # pl.col("price").pbv.pbv(volume=pl.col("volume"), window_size=window_size, bins=bins).alias("pbv")
    )
    print(expected_df)
    print(result)
    assert result.equals(expected_df)


def test_pbv_pct():
    price_col = [100, 101, 102, 103, 104, 105, 106]
    volume_col = [200, 220, 250, 240, 260, 300, 280]
    window_size = 6
    bins = 3
    df = pl.DataFrame({"price": price_col, "volume": volume_col})
    expected_df = (
        pl.DataFrame(
            {
                "price": [
                    *[
                        None,
                    ]
                    * 5,
                    [100.0, 101.6667, 103.3333],
                    [101.0, 102.6667, 104.3333],
                ],
                "volume": [
                    *[
                        None,
                    ]
                    * 5,
                    [
                        (200 + 220) / sum(volume_col[:6]),
                        (250 + 240) / sum(volume_col[:6]),
                        (260 + 300) / sum(volume_col[:6]),
                    ],
                    [
                        (220 + 250) / sum(volume_col[1:7]),
                        (240 + 260) / sum(volume_col[1:7]),
                        (280 + 300) / sum(volume_col[1:7]),
                    ],
                ],
            }
        )
        .with_columns(
            pl.col("volume").list.eval(pl.element().round(4)),
        )
        .select(pl.struct("price", "volume").alias("pbv_pct"))
    )

    result = df.select(
        pbv_pct(
            "price", "volume", window_size=window_size, bins=bins, center=False, round=4
        ).alias("pbv_pct")
    )
    assert result.equals(expected_df)


def test_pbv_top_vp():
    price_col = [100, 101, 102, 103, 104, 105, 106]
    volume_col = [200, 220, 250, 240, 260, 300, 280]
    window_size = 6
    bins = 3
    n = 2

    df = pl.DataFrame({"price": price_col, "volume": volume_col})
    expected_df = pl.DataFrame(
        {"pbv_top_vp": [None, None, None, None, None, [104.17, 102.5], [105.17, 103.5]]}
    )

    result_df = df.select(
        pbv_topn_vp(
            "price",
            "volume",
            window_size=window_size,
            bins=bins,
            n=n,
            center=True,
            round=2,
        ).alias("pbv_top_vp")
    )

    print(expected_df)
    print(result_df)
    assert result_df.equals(expected_df)


# pbv_topn_v test case
def test_pbv_top_v():
    price_col = [100, 101, 102, 103, 104, 105, 106]
    volume_col = [200, 220, 250, 240, 260, 300, 280]
    window_size = 6
    bins = 3
    n = 2

    df = pl.DataFrame({"price": price_col, "volume": volume_col})
    expected_df = pl.DataFrame(
        {"pbv_top_v": [None, None, None, None, None, [560.0, 490.0], [580.0, 500.0]]}
    )

    result_df = df.select(
        pbv_topn_v(
            "price",
            "volume",
            window_size=window_size,
            bins=bins,
            n=n,
            center=False,
            round=-1,
            pct=False,
        ).alias("pbv_top_v")
    )

    print(expected_df)
    print(result_df)
    assert result_df.equals(expected_df)
