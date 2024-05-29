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

#[polars_expr(output_type_func=price_by_volume_dtype)]
fn price_by_volume(inputs: &[Series], kwargs: PriceByVolumeKwargs) -> PolarsResult<Series> {
    let price = &inputs[0].to_float()?;
    let volume = &inputs[1].to_float()?;
    let window_size = kwargs.window_size as usize;
    let bins = kwargs.bins;
    let mut pbv = vec![];
    let mut label = vec![];
    for i in 1..(price.len() + 1) {
        // println!("i: {}", i);
        if i < (window_size) {
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
                if kwargs.center_label {
                    price_label.push(center);
                } else {
                    price_label.push(lower_bound);
                }
            }
            // println!("{:?}", volume_at_price);
            pbv.push(Some(Series::new("volume", &volume_at_price)));
            let price_label_s = if kwargs.round < 0 {
                Series::new("price", &price_label)
            } else {
                Series::new("price", &price_label).round(kwargs.round as u32)?
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
fn price_by_volume_pct(inputs: &[Series], kwargs: PriceByVolumeKwargs) -> PolarsResult<Series> {
    let price = &inputs[0].to_float()?;
    let volume = &inputs[1].to_float()?;
    let window_size = kwargs.window_size as usize;
    let bins = kwargs.bins;
    let mut pbv = vec![];
    let mut label = vec![];
    for i in 1..(price.len() + 1) {
        // println!("i: {}", i);
        if i < (window_size) {
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
                if kwargs.center_label {
                    price_label.push(center);
                } else {
                    price_label.push(lower_bound);
                }
            }
            // println!("{:?}", volume_at_price);
            let pbv_s = Series::new("volume", &volume_at_price);
            let total_volume: f64 = pbv_s.sum()?;
            let pbv_s = if kwargs.round < 0 {
                pbv_s / total_volume
            } else {
                (pbv_s / total_volume).round(kwargs.round as u32)?
            };
            pbv.push(Some(pbv_s));
            let price_label_s = if kwargs.round < 0 {
                Series::new("price", &price_label)
            } else {
                Series::new("price", &price_label).round(kwargs.round as u32)?
            };
            label.push(Some(price_label_s));
        }
    }
    let label_series = Series::new("price", &label);
    let pbv_series = Series::new("volume", &pbv);
    let out = StructChunked::new("pbv", &vec![label_series, pbv_series])?;
    Ok(out.into_series())
}
