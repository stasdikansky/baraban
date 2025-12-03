# baraban

![baraban](./assets/logo.png)

Библиотека для проведения A/B тестов и расчета размера выборки.

## Установка

```bash
pip install baraban
```

## Пример использования

```python
from baraban import ABTester

tester = ABTester()

# Расчет размера выборки
sample_size = tester.calculate_sample_size(
    metrics=['revenue', 'retention'],
    effect_sizes=[0.05, 0.1],  # 5% и 10%
    pre_experiment_data=pre_experiment_data,
    historical_data=historical_data,
    strata=['geo', 'platform'],
    alpha=0.05,
    power=0.8,
    outliers_handling_method='replace_threshold',
    outliers_threshold_quantile=0.995,
    outliers_type='upper',
    continuous_alternative='larger',
    binary_mde='absolute'
)

# Проведение A/B теста
ab_test = tester.run_abtest(
    metrics=['revenue', 'retention'],
    experiment_data=experiment_data,
    group_column='ab_group',
    groups=['control', 'test'],
    historical_data=historical_data,
    strata=['geo', 'platform'],
    alpha=0.05,
    power=0.8,
    outliers_handling_method='replace_threshold',
    outliers_threshold_quantile=0.995,
    outliers_type='upper',
    continuous_alternative='larger',
)
```

## Требования

- Python 3.8+
- pandas 1.3+
- numpy 1.20+
- scipy 1.7+
- statsmodels 0.13+
- pydantic 2.0+
- otvertka 0.1.10+

## Лицензия

MIT
