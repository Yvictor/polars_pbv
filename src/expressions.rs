#![allow(clippy::unused_unit)]
use polars::prelude::*;
// use polars::prelude::
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct PriceByVolumeKwargs {
    window_size: i32,
    bins: i32,
    center_label: bool,
    round: i32,
}

#[derive(Deserialize)]
pub struct PriceByVolumeTopNKwargs {
    window_size: i32,
    bins: i32,
    n: usize,
    center_label: bool,
    round: i32,
}

// fn price_by_volume_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
//     let field = Field::new(
//         "pbv",
//         DataType::List(Box::new(input_fields[1].dtype.clone())),
//     );
//     Ok(field)
// }

fn price_by_volume_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    let field_price = Field::new(
        "price",
        DataType::List(Box::new(input_fields[0].dtype.clone())),
    );
    let field_volume = Field::new(
        "volume",
        DataType::List(Box::new(input_fields[1].dtype.clone())),
    );
    let v: Vec<Field> = vec![field_price, field_volume];
    Ok(Field::new("pbv", DataType::Struct(v)))
}

fn price_by_volume(
    price: &Series,
    volume: &Series,
    window_size: i32,
    bins: i32,
    center_label: bool,
    round: i32,
    pct: bool,
) -> PolarsResult<Series> {
    let window_size = window_size as usize;
    let mut pbv = vec![];
    let mut label = vec![];
    for i in 1..(price.len() + 1) {
        // println!("i: {}", i);
        if i < window_size {
            pbv.push(None);
            label.push(None);
        } else {
            let mut volume_at_price = vec![];
            let mut price_label = vec![];
            let start = (i - window_size) as i64;
            let window_price = price.slice(start, window_size);
            let window_volume = volume.slice(start, window_size);
            let max_price: f64 = window_price.max()?.unwrap();
            let min_price: f64 = window_price.min()?.unwrap();
            let range = max_price - min_price;
            let interval = range / bins as f64;
            for n in 0..bins {
                let lower_bound = min_price + n as f64 * interval;
                let upper_bound = min_price + (n + 1) as f64 * interval;
                let center = (lower_bound + upper_bound) / 2.0;
                if n == bins - 1 {
                    // println!("start: {}, lower: {}", start, lower_bound);
                    let v: f64 = window_volume
                        .filter(&window_price.gt_eq(lower_bound)?)?
                        .sum()?;
                    volume_at_price.push(v);
                } else {
                    // println!(
                    //     "start: {}, lower: {}, upper: {}",
                    //     start, lower_bound, upper_bound
                    // );
                    let mask = window_price.gt_eq(lower_bound)? & window_price.lt(upper_bound)?;
                    let v = window_volume.filter(&mask)?.sum()?;
                    volume_at_price.push(v);
                }
                if center_label {
                    price_label.push(center);
                } else {
                    price_label.push(lower_bound);
                }
            }
            let pbv_s = Series::new("volume", &volume_at_price);
            let pbv_s = if pct {
                let total_volume: f64 = pbv_s.sum()?;
                if round < 0 {
                    pbv_s / total_volume
                } else {
                    (pbv_s / total_volume).round(round as u32)?
                }
            } else {
                if round < 0 {
                    pbv_s
                } else {
                    pbv_s.round(round as u32)?
                }
            };
            pbv.push(Some(pbv_s));
            let price_label_s = if round < 0 {
                Series::new("price", &price_label)
            } else {
                Series::new("price", &price_label).round(round as u32)?
            };
            label.push(Some(price_label_s));
        }
    }
    let label_series = Series::new("price", &label);
    let pbv_series = Series::new("volume", &pbv);
    let out = StructChunked::new("pbv", &vec![label_series, pbv_series])?;
    Ok(out.into_series())
}

#[polars_expr(output_type_func=price_by_volume_dtype)]
fn pbv(inputs: &[Series], kwargs: PriceByVolumeKwargs) -> PolarsResult<Series> {
    let price = &inputs[0].to_float()?;
    let volume = &inputs[1].to_float()?;
    price_by_volume(
        price,
        volume,
        kwargs.window_size,
        kwargs.bins,
        kwargs.center_label,
        kwargs.round,
        false,
    )
}

#[polars_expr(output_type_func=price_by_volume_dtype)]
fn pbv_pct(inputs: &[Series], kwargs: PriceByVolumeKwargs) -> PolarsResult<Series> {
    let price = &inputs[0].to_float()?;
    let volume = &inputs[1].to_float()?;
    price_by_volume(
        price,
        volume,
        kwargs.window_size,
        kwargs.bins,
        kwargs.center_label,
        kwargs.round,
        true,
    )
}

fn price_by_volume_topn_volume_price_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = Field::new(
        "pbv_topn_vp",
        DataType::List(Box::new(input_fields[0].dtype.clone())),
    );
    Ok(field)
}

#[polars_expr(output_type_func=price_by_volume_topn_volume_price_dtype)]
fn pbv_topn_vp(inputs: &[Series], kwargs: PriceByVolumeTopNKwargs) -> PolarsResult<Series> {
    let price = &inputs[0].to_float()?;
    let volume = &inputs[1].to_float()?;
    let window_size = kwargs.window_size as usize;
    let mut pbv_topn = vec![];
    for i in 1..(price.len() + 1) {
        if i < window_size {
            pbv_topn.push(None);
        } else {
            let mut volume_at_price = vec![];
            let mut price_label = vec![];
            let start = (i - window_size) as i64;
            let window_price = price.slice(start, window_size);
            let window_volume = volume.slice(start, window_size);
            let max_price: f64 = window_price.max()?.unwrap();
            let min_price: f64 = window_price.min()?.unwrap();
            let range = max_price - min_price;
            let interval = range / kwargs.bins as f64;
            for n in 0..kwargs.bins {
                let lower_bound = min_price + n as f64 * interval;
                let upper_bound = min_price + (n + 1) as f64 * interval;
                let center = (lower_bound + upper_bound) / 2.0;
                if n == kwargs.bins - 1 {
                    let v: f64 = window_volume
                        .filter(&window_price.gt_eq(lower_bound)?)?
                        .sum()?;
                    volume_at_price.push(v);
                } else {
                    let mask = window_price.gt_eq(lower_bound)? & window_price.lt(upper_bound)?;
                    let v = window_volume.filter(&mask)?.sum()?;
                    volume_at_price.push(v);
                }
                let label = if kwargs.center_label {
                    center
                } else {
                    lower_bound
                };
                price_label.push(label);
            }
            let price_label_s = Series::new("price", &price_label);
            let price_label_s_round = price_label_s.round(kwargs.round as u32)?;
            let price_label_s = if kwargs.round < 0 {
                price_label_s.f64()?
            } else {
                price_label_s_round.f64()?
            };
            let pbv_s = Series::new("volume", &volume_at_price);
            let pbv_s_idx_sort =
                pbv_s.arg_sort(SortOptions::default().with_order_descending(true));
            
            let pbv_topn_s = pbv_s_idx_sort.slice(0, kwargs.n as usize)
                .iter()
                .map(|opt_idx| {
                     match opt_idx {
                         Some(idx) => Some(price_label_s.get(idx as usize).unwrap()),
                         None => None, // Return None for out-of-bounds indices
                     }
                 })
                .collect::<Vec<Option<f64>>>();

            pbv_topn.push(Some(Series::new("pbv_topn", &pbv_topn_s)));
            // pbv.push(Some(pbv_s));
            
            // label.push(Some(price_label_s));
        }
    }
    Ok(Series::new("pbv_topn", pbv_topn))
}
