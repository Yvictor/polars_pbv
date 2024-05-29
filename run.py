import polars as pl
import polars_pbv as pl_pbv

price_col = [100, 101, 102, 103, 104, 105, 106, ]#107, 108, 109]
volume_col = [200, 220, 250, 240, 260, 300, 280, ]# 270, 310, 330]
window_size = 6#5.0
bins = 3
df = pl.DataFrame({
    'price': price_col,
    'volume': volume_col
})
df_result = df.with_columns(
    pl_pbv.pbv("price", "volume", window_size=window_size, bins=bins, center=True, round=2).alias("pbv"),
    pl_pbv.pbv_pct("price", "volume", window_size=window_size, bins=bins, center=True, round=4).alias("pbv_pct"),
    pl_pbv.pbv_topn_vp("price", "volume", window_size=window_size, bins=bins, n=2, center=True, round=2).alias("pbv_topn_vp"),
    pl_pbv.pbv_topn_vp("price", "volume", window_size=window_size, bins=bins, n=1, center=True, round=2).list.get(0).alias("pbv_top1_vp"),
    pl_pbv.pbv_topn_v("price", "volume", window_size=window_size, bins=bins, n=2, center=True, round=2).alias("pbv_topn_v"),
)
print(df_result)

