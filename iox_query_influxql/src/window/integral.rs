//! InfluxQL `integral()` window function: cumulative integral of values over time.
//! integral(field) or integral(field, duration) => sum of (value_i * (t_i - t_{i-1})) in duration units.

use crate::{error, NUMERICS};
use arrow::array::{Array, ArrayRef};
use arrow::datatypes::{DataType, Field, IntervalUnit::MonthDayNano, TimeUnit};
use datafusion::common::{Result, ScalarValue};
use datafusion::logical_expr::function::{
    ExpressionArgs, PartitionEvaluatorArgs, WindowUDFFieldArgs,
};
use datafusion::logical_expr::{
    PartitionEvaluator, Signature, TypeSignature, Volatility, WindowUDFImpl, TIMEZONE_WILDCARD,
};
use datafusion::physical_expr::PhysicalExpr;
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct IntegralUDWF {
    signature: Signature,
}

impl IntegralUDWF {
    pub(super) fn new() -> Self {
        Self {
            signature: Signature::one_of(
                NUMERICS
                    .iter()
                    .flat_map(|dt| {
                        [
                            TypeSignature::Exact(vec![
                                dt.clone(),
                                DataType::Interval(MonthDayNano),
                                DataType::Timestamp(TimeUnit::Nanosecond, None),
                            ]),
                            TypeSignature::Exact(vec![
                                dt.clone(),
                                DataType::Interval(MonthDayNano),
                                DataType::Timestamp(
                                    TimeUnit::Nanosecond,
                                    Some(TIMEZONE_WILDCARD.into()),
                                ),
                            ]),
                        ]
                    })
                    .collect(),
                Volatility::Immutable,
            ),
        }
    }
}

impl WindowUDFImpl for IntegralUDWF {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "integral"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn field(&self, _field_args: WindowUDFFieldArgs<'_>) -> Result<Field> {
        Ok(Field::new("integral", DataType::Float64, true))
    }

    fn expressions(&self, expr_args: ExpressionArgs<'_>) -> Vec<Arc<dyn PhysicalExpr>> {
        expr_args.input_exprs().into()
    }

    fn partition_evaluator(
        &self,
        _args: PartitionEvaluatorArgs<'_>,
    ) -> Result<Box<dyn PartitionEvaluator>> {
        Ok(Box::new(IntegralPartitionEvaluator {}))
    }
}

/// PartitionEvaluator: running integral = sum of (value_i * delta_t_i / unit_ns).
#[derive(Debug)]
struct IntegralPartitionEvaluator {}

impl PartitionEvaluator for IntegralPartitionEvaluator {
    fn evaluate_all(&mut self, values: &[ArrayRef], _num_rows: usize) -> Result<Arc<dyn Array>> {
        assert_eq!(values.len(), 3);

        let array = Arc::clone(&values[0]);
        let times = Arc::clone(&values[2]);
        let unit = ScalarValue::try_from_array(&values[1], 0)?;

        let mut idx: usize = 0;
        let mut last_time: ScalarValue = times.data_type().try_into()?;
        let mut running: f64 = 0.0;
        let mut integral: Vec<ScalarValue> = vec![];

        while idx < array.len() {
            let v = ScalarValue::try_from_array(&array, idx)?;
            let t = ScalarValue::try_from_array(&times, idx)?;
            if v.is_null() {
                integral.push(ScalarValue::Float64(None));
                idx += 1;
            } else {
                integral.push(ScalarValue::Float64(Some(0.0)));
                last_time = t;
                idx += 1;
                break;
            }
        }
        while idx < array.len() {
            let v = ScalarValue::try_from_array(&array, idx)?;
            let t = ScalarValue::try_from_array(&times, idx)?;
            if v.is_null() {
                integral.push(ScalarValue::Float64(None));
            } else {
                let val_f64 = value_to_f64(&v)?;
                let dt_ns = delta_time_ns(&t, &last_time)?;
                let unit_ns = interval_nanoseconds(&unit)?;
                running += val_f64 * dt_ns / unit_ns;
                integral.push(ScalarValue::Float64(Some(running)));
                last_time = t;
            }
            idx += 1;
        }
        Ok(Arc::new(ScalarValue::iter_to_array(integral)?))
    }

    fn uses_window_frame(&self) -> bool {
        false
    }

    fn include_rank(&self) -> bool {
        false
    }
}

fn value_to_f64(v: &ScalarValue) -> Result<f64> {
    match v {
        ScalarValue::Float64(Some(x)) => Ok(*x),
        ScalarValue::Int64(Some(x)) => Ok(*x as f64),
        ScalarValue::UInt64(Some(x)) => Ok(*x as f64),
        _ => error::internal("integral attempted on unsupported value type"),
    }
}

fn delta_time_ns(curr: &ScalarValue, prev: &ScalarValue) -> Result<f64> {
    if let (
        ScalarValue::TimestampNanosecond(Some(c), _),
        ScalarValue::TimestampNanosecond(Some(p), _),
    ) = (curr, prev)
    {
        Ok((*c - *p) as f64)
    } else {
        error::internal("integral: unsupported timestamp types")
    }
}

fn interval_nanoseconds(unit: &ScalarValue) -> Result<f64> {
    if let ScalarValue::IntervalMonthDayNano(Some(u)) = unit {
        Ok(u.nanoseconds as f64)
    } else {
        error::internal("integral: unit must be IntervalMonthDayNano")
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, Float64Array, TimestampNanosecondArray};
    use arrow::datatypes::IntervalMonthDayNano;
    use datafusion::common::ScalarValue;
    use std::sync::Arc;

    use super::*;

    const NS_PER_S: i64 = 1_000_000_000;

    fn unit_array_1s(len: usize) -> ArrayRef {
        let unit = IntervalMonthDayNano::new(0, 0, NS_PER_S);
        Arc::new(arrow::array::IntervalMonthDayNanoArray::from(
            (0..len).map(|_| Some(unit)).collect::<Vec<_>>(),
        ))
    }

    #[test]
    fn test_integral_evaluate_all() {
        // integral = sum of (value_i * (t_i - t_{i-1}) / unit). First row gets 0.0 (no prior interval).
        // values [1,2,3] at t=0,1s,2s, unit=1s => row0=0, row1=0+2*1=2, row2=2+3*1=5
        let values: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0), Some(3.0)]));
        let times: ArrayRef = Arc::new(TimestampNanosecondArray::from(vec![
            Some(0),
            Some(NS_PER_S),
            Some(2 * NS_PER_S),
        ]));

        let mut evaluator = IntegralPartitionEvaluator {};
        let inputs = vec![values, unit_array_1s(3), times];
        let out = evaluator.evaluate_all(&inputs, 3).unwrap();

        let out_f64 = out.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(out_f64.len(), 3);
        assert!(!out_f64.is_null(0), "first row integral must be 0.0, not null");
        assert_eq!(out_f64.value(0), 0.0);
        assert_eq!(out_f64.value(1), 2.0);
        assert_eq!(out_f64.value(2), 5.0);
    }

    #[test]
    fn test_integral_evaluate_all_with_leading_null() {
        // Leading nulls stay null; first non-null gets 0.0; then cumulative: null, 1.0, 2.0 at t=0,1s,2s => null, 0, 2
        let values: ArrayRef = Arc::new(Float64Array::from(vec![None, Some(1.0), Some(2.0)]));
        let times: ArrayRef = Arc::new(TimestampNanosecondArray::from(vec![
            Some(0),
            Some(NS_PER_S),
            Some(2 * NS_PER_S),
        ]));

        let mut evaluator = IntegralPartitionEvaluator {};
        let inputs = vec![values, unit_array_1s(3), times];
        let out = evaluator.evaluate_all(&inputs, 3).unwrap();

        let out_f64 = out.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(out_f64.len(), 3);
        assert!(out_f64.is_null(0));
        assert_eq!(out_f64.value(1), 0.0);
        assert_eq!(out_f64.value(2), 2.0);
    }
}
