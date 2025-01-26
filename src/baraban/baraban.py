from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as ss
from otvertka import handle_outliers
from pydantic import BaseModel, Field, field_validator
from statsmodels.stats.power import tt_ind_solve_power
from tqdm import tqdm


class OutlierHandlingMethod(str, Enum):
    """Available methods for handling outliers"""
    DROP = "drop"
    REPLACE_THRESHOLD = "replace_threshold"
    REPLACE_MEDIAN = "replace_median"

class OutlierType(str, Enum):
    """Types of outliers to handle"""
    UPPER = "upper"
    LOWER = "lower"
    TWO_SIDED = "two-sided"

class OutlierConfig(BaseModel):
    """Configuration for outlier handling"""
    handling_method: Optional[OutlierHandlingMethod] = Field(
        None, 
        description="Method for handling outliers"
    )
    threshold_quantile: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Quantile threshold for outlier detection"
    )
    type: Optional[OutlierType] = Field(
        None,
        description="Type of outliers to handle"
    )

    @field_validator("threshold_quantile")
    @classmethod
    def validate_threshold(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v <= 0 or v >= 1):
            raise ValueError("Threshold quantile must be between 0 and 1")
        return v

class BaseConfig(BaseModel):
    """Base configuration for A/B testing"""
    metrics: List[str] = Field(..., min_items=1, description="List of metric names to analyze")
    alpha: float = Field(
        0.05, 
        gt=0, 
        lt=1, 
        description="Significance level (Type I error rate)"
    )
    power: float = Field(
        0.8, 
        gt=0, 
        lt=1, 
        description="Statistical power (1 - Type II error rate)"
    )
    historical_data: Optional[pd.DataFrame] = None
    strata: Optional[List[str]] = Field(
        None,
        min_items=1,
        description="List of column names to use for stratification"
    )
    outliers: Optional[OutlierConfig] = Field(
        None,
        description="Configuration for outlier handling"
    )

    @field_validator("strata")
    @classmethod
    def validate_strata(cls, v: Optional[List[str]], info) -> Optional[List[str]]:
        if v is not None and info.data.get("historical_data") is None:
            raise ValueError("Historical data is required when using stratification")
        if v is not None and info.data.get("historical_data") is not None:
            missing_cols = [col for col in v if col not in info.data["historical_data"].columns]
            if missing_cols:
                raise ValueError(f"Stratification columns {missing_cols} not found in historical data")
        return v

    class Config:
        arbitrary_types_allowed = True

class SampleSizeConfig(BaseConfig):
    """Configuration for sample size calculation"""
    effect_sizes: List[float] = Field(
        ..., 
        min_items=1,
        description="List of relative changes to detect (e.g. [0.01, 0.02] for 1% and 2%)"
    )
    pre_experiment_data: pd.DataFrame = Field(
        ...,
        description="DataFrame with pre-experiment data"
    )

    @field_validator("effect_sizes")
    @classmethod
    def validate_effect_sizes(cls, v: List[float]) -> List[float]:
        if any(size <= 0 for size in v):
            raise ValueError("Effect sizes must be positive")
        return v

    @field_validator("pre_experiment_data")
    @classmethod
    def validate_metrics_in_data(cls, v: pd.DataFrame, info) -> pd.DataFrame:
        if "metrics" in info.data:
            missing_metrics = [m for m in info.data["metrics"] if m not in v.columns]
            if missing_metrics:
                raise ValueError(f"Metrics {missing_metrics} not found in pre_experiment_data")
        return v

class ABTestConfig(BaseConfig):
    """Configuration for A/B test execution"""
    experiment_data: pd.DataFrame = Field(
        ...,
        description="DataFrame with experiment data"
    )
    group_column: str = Field(
        "ab_group",
        description="Name of column containing group labels"
    )
    groups: Optional[List[str]] = Field(
        None,
        min_items=2,
        max_items=2,
        description="List of group names [control, test]"
    )
    boot_iterations: int = Field(
        10000,
        gt=0,
        description="Number of bootstrap iterations"
    )

    @field_validator("experiment_data")
    @classmethod
    def validate_data(cls, v: pd.DataFrame, info) -> pd.DataFrame:
        if "metrics" in info.data:
            missing_metrics = [m for m in info.data["metrics"] if m not in v.columns]
            if missing_metrics:
                raise ValueError(f"Metrics {missing_metrics} not found in experiment_data")
        
        if "group_column" in info.data and info.data["group_column"] not in v.columns:
            raise ValueError(f"Group column {info.data['group_column']} not found in experiment_data")
        
        return v

    @field_validator("groups")
    @classmethod
    def validate_groups(cls, v: Optional[List[str]], info) -> Optional[List[str]]:
        if v is not None:
            if len(v) != 2:
                raise ValueError("The length of groups must be 2")
            if "experiment_data" in info.data and "group_column" in info.data:
                unique_groups = info.data["experiment_data"][info.data["group_column"]].unique()
                missing_groups = [g for g in v if g not in unique_groups]
                if missing_groups:
                    raise ValueError(f"Groups {missing_groups} not found in experiment_data")
        return v

class ABTestStrategy(ABC):
    """Abstract base class for A/B testing strategies.

    This class defines the interface for different A/B testing strategies
    and provides common functionality for outlier handling.

    Attributes
    ----------
    config : BaseConfig
        Configuration object specific to the strategy
    weights : dict, optional
        Dictionary of stratum weights for stratified analysis
    """
    
    def _should_handle_outliers(self) -> bool:
        """Check if outlier handling is configured.

        Returns
        -------
        bool
            True if any outlier handling parameters are set, False otherwise
        """
        return any([
            self.config.outliers.handling_method is not None,
            self.config.outliers.threshold_quantile is not None,
            self.config.outliers.type is not None
        ])
    
    def _handle_outliers(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """Handle outliers in the specified metric column.

        Applies configured outlier handling method to the metric column.
        Skips binary metrics (containing only 0/1 values).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        metric : str
            Name of column to handle outliers in

        Returns
        -------
        pd.DataFrame
            DataFrame with outliers handled according to configuration
        """
        if not self._should_handle_outliers():
            return df
            
        # Skip outlier handling for binary metrics
        if set(df[metric].unique()).issubset({0, 1}):
            return df
            
        handling_method = self.config.outliers.handling_method.value if self.config.outliers.handling_method else "replace_threshold"
        threshold_quantile = self.config.outliers.threshold_quantile or 0.995
        outlier_type = self.config.outliers.type.value if self.config.outliers.type else "upper"
        
        return handle_outliers(
            df=df,
            target_column=metric,
            threshold_quantile=threshold_quantile,
            handling_method=handling_method,
            outlier_type=outlier_type
        )
    
    @abstractmethod
    def execute(self) -> pd.DataFrame:
        """Execute the A/B testing strategy.

        Returns
        -------
        pd.DataFrame
            DataFrame containing analysis results
        """
        pass

class SampleSizeCalculator(ABTestStrategy):
    """Strategy for calculating required sample size.

    This class implements sample size calculation for both binary and continuous metrics,
    with optional stratification and outlier handling.

    Attributes
    ----------
    config : SampleSizeConfig
        Configuration object containing calculation parameters
    weights : dict, optional
        Dictionary of stratum weights for stratified analysis
    """
    def __init__(self, config: SampleSizeConfig):
        self.config = config
        self.weights: Optional[dict] = None

    def _is_binary_metric(self, df: pd.DataFrame, metric: str) -> bool:
        """Check if metric contains only binary (0/1) values.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        metric : str
            Name of column to check

        Returns
        -------
        bool
            True if metric contains only 0/1 values, False otherwise
        """
        unique_values = df[metric].nunique()
        return unique_values == 2 and set(df[metric].unique()).issubset({0, 1})

    def _calc_sample_size_for_metric(
        self, df: pd.DataFrame, metric: str, is_stratified: bool = False
    ) -> pd.DataFrame:
        """Calculate required sample size for a single metric.

        For binary metrics, uses Evan Miller's method.
        For continuous metrics, uses power analysis with Cohen's d.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        metric : str
            Name of metric to analyze
        is_stratified : bool, optional
            Whether to use stratified analysis, by default False

        Returns
        -------
        pd.DataFrame
            DataFrame with required sample sizes for each effect size
        """
        data = []
        
        if self._is_binary_metric(df, metric):
            curr_conversion = df[metric].sum() / df[metric].count()
            
            row = []
            for effect_size in self.config.effect_sizes:
                n = calc_evan_miller_sample_size(
                    alpha=self.config.alpha, 
                    power=self.config.power, 
                    base_rate=curr_conversion, 
                    pct_mde=effect_size
                )
                row.append(n // 100 * 100)
            data.append(row)
        else:
            metric_mean = df[metric].mean()
            std_dev = df[metric].std()
            
            if is_stratified:
                metric_mean = calc_stratified_mean(
                    df=df, 
                    strat="strat", 
                    metric=metric, 
                    weights=self.weights
                )
                std_dev = np.sqrt(calc_stratified_variance(
                    df=df, 
                    strat="strat", 
                    metric=metric, 
                    weights=self.weights
                ))

            row = []
            for effect_size in self.config.effect_sizes:
                absolute_change = metric_mean * effect_size
                cohen_d = absolute_change / std_dev
                sample_size = tt_ind_solve_power(
                    effect_size=cohen_d,
                    alpha=self.config.alpha,
                    power=self.config.power,
                    alternative="two-sided",
                )
                row.append(round(float(sample_size)) // 100 * 100)
            data.append(row)

        columns = [f"{change * 100:.1f}%" for change in self.config.effect_sizes]
        df_sample_size = pd.DataFrame(data, columns=columns)
        df_sample_size.index = [metric.replace("_", " ").title()]
        
        return df_sample_size

    def execute(self) -> pd.DataFrame:
        """Execute sample size calculation for all metrics.

        Process:
            1. Handle outliers if configured (skip for binary metrics)
            2. Apply stratification if historical data and strata are provided
            3. Calculate sample size for each metric
            4. Combine and format results

        Returns
        -------
        pd.DataFrame
            Styled DataFrame with required sample sizes for each metric and effect size
        """
        results = {}
        df = self.config.pre_experiment_data.copy()
        
        if self.config.historical_data is not None and self.config.strata:
            # Prepare data for stratification
            df_historical = self.config.historical_data.copy()
            
            # Handle outliers before stratification
            for metric in self.config.metrics:
                if not self._is_binary_metric(df, metric):  # skip for binary metrics
                    df = self._handle_outliers(df, metric)
                    df_historical = self._handle_outliers(df_historical, metric)
            
            # Apply stratification
            for strat in self.config.strata:
                unique_values = (
                    df_historical[strat]
                    .value_counts()
                    .to_frame()
                    .reset_index()[strat]
                    .tolist()
                )
                df_historical[strat] = df_historical[strat].apply(
                    lambda x: x if x in unique_values else f"other_{strat}"
                )
                df[strat] = df[strat].apply(
                    lambda x: x if x in unique_values else f"other_{strat}"
                )

            df_historical["strat"] = df_historical[self.config.strata].agg(" | ".join, axis=1)
            df["strat"] = df[self.config.strata].agg(" | ".join, axis=1)
            
            # Calculate stratification weights
            strat_weights = (
                df_historical["strat"].value_counts() / df_historical["strat"].count()
            )
            self.weights = strat_weights.to_dict()
            
            is_stratified = True
        else:
            # If no stratification, just handle outliers
            for metric in self.config.metrics:
                if not self._is_binary_metric(df, metric):
                    df = self._handle_outliers(df, metric)
            is_stratified = False

        for metric in self.config.metrics:
            results[metric] = self._calc_sample_size_for_metric(df, metric, is_stratified)

        df_result = pd.concat([results[metric] for metric in self.config.metrics])
        
        # Add styling
        return df_result.style.set_caption(
            f"Sample Sizes For Significance Level {int((1 - self.config.alpha) * 100)}%"
        ).format(precision=0)

class ABTestCalculator(ABTestStrategy):
    """Strategy for executing A/B test analysis.

    This class implements A/B test analysis using bootstrap resampling,
    with optional stratification and outlier handling.

    Attributes
    ----------
    config : ABTestConfig
        Configuration object containing test parameters
    weights : dict, optional
        Dictionary of stratum weights for stratified analysis
    """
    def __init__(self, config: ABTestConfig):
        self.config = config
        self.weights: Optional[dict] = None

    def _is_binary_metric(self, df: pd.DataFrame, metric: str) -> bool:
        """Check if metric contains only binary (0/1) values.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        metric : str
            Name of column to check

        Returns
        -------
        bool
            True if metric contains only 0/1 values, False otherwise
        """
        unique_values = df[metric].nunique()
        return unique_values == 2 and set(df[metric].unique()).issubset({0, 1})

    def _run_bootstrap(
        self, 
        df: pd.DataFrame, 
        metric: str, 
        is_stratified: bool = False
    ) -> pd.DataFrame:
        """Run bootstrap analysis for a single metric.

        Performs parallel bootstrap resampling to estimate the sampling distribution
        of the difference between control and test groups.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        metric : str
            Name of metric to analyze
        is_stratified : bool, optional
            Whether to use stratified sampling, by default False

        Returns
        -------
        pd.DataFrame
            DataFrame with test results including:
                - P-value
                - Control and test means
                - Relative difference
                - Sample sizes
                - Confidence bounds
                - Statistical significance verdict

        Raises
        ------
        ValueError
            If number of groups is not exactly 2
        """
        if len(self.config.groups) != 2:
            raise ValueError("The length of groups must be 2")

        control_group = self.config.groups[0]
        test_group = self.config.groups[1]

        if is_stratified:
            control = df.query(f"{self.config.group_column} == @control_group")[[metric, "strat"]]
            test = df.query(f"{self.config.group_column} == @test_group")[[metric, "strat"]]
        else:
            control = df.query(f"{self.config.group_column} == @control_group")[metric].values
            test = df.query(f"{self.config.group_column} == @test_group")[metric].values
            
        boot_len = max(len(control), len(test))
        boot_data = np.zeros(self.config.boot_iterations)
        control_means = np.zeros(self.config.boot_iterations)
        test_means = np.zeros(self.config.boot_iterations)

        # Parallel execution of bootstrap iterations
        with ProcessPoolExecutor() as executor:
            futures = []
            for _ in range(self.config.boot_iterations):
                future = executor.submit(
                    run_bootstrap_iteration,
                    control=control,
                    test=test,
                    boot_len=boot_len,
                    is_stratified=is_stratified,
                    strat="strat" if is_stratified else None,
                    weights=self.weights if is_stratified else None,
                    metric=metric if is_stratified else None,
                )
                futures.append(future)
            
            for i, future in enumerate(futures):
                control_stat_value, test_stat_value, diff = future.result()
                boot_data[i] = diff
                control_means[i] = control_stat_value
                test_means[i] = test_stat_value

        p_value = round(
            min(np.sum(boot_data >= 0), np.sum(boot_data <= 0))
            * 2
            / len(boot_data),
            3
        )

        control_mean = round(np.mean(control[metric] if is_stratified else control), 3)
        test_mean = round(np.mean(test[metric] if is_stratified else test), 3)

        if self._is_binary_metric(df, metric):
            control_conversion = round(control[metric].sum() / control[metric].count() * 100, 3) if is_stratified else round(np.sum(control) / len(control) * 100, 3)
            test_conversion = round(test[metric].sum() / test[metric].count() * 100, 3) if is_stratified else round(np.sum(test) / len(test) * 100, 3)
            diff_percent = round(test_conversion - control_conversion, 3)
        else:
            diff_percent = round((test_mean - control_mean) / control_mean * 100, 3) if control_mean != 0 else None

        diff_percent = f"{diff_percent}%"

        lower_bound = round(np.quantile(boot_data, self.config.alpha / 2), 3)
        upper_bound = round(np.quantile(boot_data, 1 - self.config.alpha / 2), 3)

        result = {
            "Control Name": control_group,
            "Test Name": test_group,
            "P-Value": p_value,
            "Control Mean": control_mean,
            "Test Mean": test_mean,
            "Diff (%)": diff_percent,
            "Control Installs": int(len(control)),
            "Test Installs": int(len(test)),
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Verdict": "Significant" if p_value < self.config.alpha else "Non-Significant"
        }

        df_result = pd.DataFrame([result])
        df_result.index = [metric.replace("_", " ").title()]
        
        return df_result

    def execute(self) -> pd.DataFrame:
        """Execute A/B test analysis for all metrics.

        Process:
            1. Handle outliers if configured (skip for binary metrics)
            2. Apply stratification if historical data and strata are provided
            3. Run parallel bootstrap analysis for each metric
            4. Combine and format results with styling

        Returns
        -------
        pd.DataFrame
            Styled DataFrame with test results for each metric
        """
        df = self.config.experiment_data.copy()
        
        if self.config.historical_data is not None and self.config.strata:
            # Prepare data for stratification
            df_historical = self.config.historical_data.copy()
            
            # Handle outliers before stratification
            for metric in self.config.metrics:
                if not self._is_binary_metric(df, metric):  # skip for binary metrics
                    df = self._handle_outliers(df, metric)
                    df_historical = self._handle_outliers(df_historical, metric)
            
            # Apply stratification
            for strat in self.config.strata:
                unique_values = (
                    df_historical[strat]
                    .value_counts()
                    .to_frame()
                    .reset_index()[strat]
                    .tolist()
                )
                df_historical[strat] = df_historical[strat].apply(
                    lambda x: x if x in unique_values else f"other_{strat}"
                )
                df[strat] = df[strat].apply(
                    lambda x: x if x in unique_values else f"other_{strat}"
                )

            df_historical["strat"] = df_historical[self.config.strata].agg(" | ".join, axis=1)
            df["strat"] = df[self.config.strata].agg(" | ".join, axis=1)
            
            # Calculate stratification weights
            strat_weights = (
                df_historical["strat"].value_counts() / df_historical["strat"].count()
            )
            self.weights = strat_weights.to_dict()
            
            is_stratified = True
        else:
            # If no stratification, just handle outliers
            for metric in self.config.metrics:
                if not self._is_binary_metric(df, metric):
                    df = self._handle_outliers(df, metric)
            is_stratified = False

        # Parallel execution of bootstrap for different metrics
        with ProcessPoolExecutor() as executor:
            futures = {
                metric: executor.submit(self._run_bootstrap, df, metric, is_stratified)
                for metric in self.config.metrics
            }
            results = {
                metric: future.result()
                for metric, future in tqdm(futures.items(), desc="Processing metrics")
            }

        df_result = pd.concat([results[metric] for metric in self.config.metrics])
        
        # Add styling
        def highlight_significant(row):
            color = 'background-color: #90EE90; color: black' if row['Verdict'] == 'Significant' else ''
            return [color] * len(row)
        
        return df_result.style.set_caption(
            f"AB Test Results For Significance Level {int((1 - self.config.alpha) * 100)}%"
        ).apply(highlight_significant, axis=1).format({
            'P-Value': '{:.3f}',
            'Control Mean': '{:.3f}',
            'Test Mean': '{:.3f}',
            'Diff (%)': lambda x: f"{float(x.strip('%')):.2f}%",
            'Lower Bound': '{:.3f}',
            'Upper Bound': '{:.3f}',
            'Control Installs': '{:.0f}',
            'Test Installs': '{:.0f}'
        })

class ABTestBuilder:
    """Builder for creating A/B test configurations.

    This class provides methods to create properly configured SampleSizeConfig
    and ABTestConfig objects from raw parameters.
    """

    def create_sample_size_config(
        self,
        metrics: List[str],
        effect_sizes: List[float],
        pre_experiment_data: pd.DataFrame,
        alpha: float = 0.05,
        power: float = 0.8,
        historical_data: Optional[pd.DataFrame] = None,
        strata: Optional[List[str]] = None,
        outliers_handling_method: Optional[str] = None,
        outliers_threshold_quantile: Optional[float] = None,
        outliers_type: Optional[str] = None,
    ) -> SampleSizeConfig:
        """Create configuration for sample size calculation.

        Parameters
        ----------
        metrics : List[str]
            List of metric names to analyze
        effect_sizes : List[float]
            List of relative changes to detect (e.g. [0.01, 0.02] for 1% and 2%)
        pre_experiment_data : pd.DataFrame
            DataFrame with pre-experiment data
        alpha : float, optional
            Significance level (Type I error rate), by default 0.05
        power : float, optional
            Statistical power (1 - Type II error rate), by default 0.8
        historical_data : pd.DataFrame, optional
            DataFrame with historical data for stratification
        strata : List[str], optional
            List of column names to use for stratification
        outliers_handling_method : str, optional
            Method for handling outliers ('remove' or 'replace_threshold')
        outliers_threshold_quantile : float, optional
            Quantile threshold for outlier detection
        outliers_type : str, optional
            Type of outliers to handle ('upper', 'lower', or 'two-sided')

        Returns
        -------
        SampleSizeConfig
            Validated SampleSizeConfig object
        """
        return SampleSizeConfig(
            metrics=metrics,
            effect_sizes=effect_sizes,
            pre_experiment_data=pre_experiment_data,
            alpha=alpha,
            power=power,
            historical_data=historical_data,
            strata=strata,
            outliers=OutlierConfig(
                handling_method=outliers_handling_method,
                threshold_quantile=outliers_threshold_quantile,
                type=outliers_type
            ),
        )

    def create_abtest_config(
        self,
        metrics: List[str],
        experiment_data: pd.DataFrame,
        group_column: str = "ab_group",
        groups: Optional[List[str]] = None,
        alpha: float = 0.05,
        power: float = 0.8,
        historical_data: Optional[pd.DataFrame] = None,
        strata: Optional[List[str]] = None,
        outliers_handling_method: Optional[str] = None,
        outliers_threshold_quantile: Optional[float] = None,
        outliers_type: Optional[str] = None,
        boot_iterations: int = 10000,
    ) -> ABTestConfig:
        """Create configuration for A/B test execution.

        Parameters
        ----------
        metrics : List[str]
            List of metric names to analyze
        experiment_data : pd.DataFrame
            DataFrame with experiment data
        group_column : str, optional
            Name of group column, by default 'ab_group'
        groups : List[str], optional
            List of group names [control, test]
        alpha : float, optional
            Significance level (Type I error rate), by default 0.05
        power : float, optional
            Statistical power (1 - Type II error rate), by default 0.8
        historical_data : pd.DataFrame, optional
            DataFrame with historical data for stratification
        strata : List[str], optional
            List of column names to use for stratification
        outliers_handling_method : str, optional
            Method for handling outliers ('remove' or 'replace_threshold')
        outliers_threshold_quantile : float, optional
            Quantile threshold for outlier detection
        outliers_type : str, optional
            Type of outliers to handle ('upper', 'lower', or 'two-sided')
        boot_iterations : int, optional
            Number of bootstrap iterations, by default 10000

        Returns
        -------
        ABTestConfig
            Validated ABTestConfig object
        """
        return ABTestConfig(
            metrics=metrics,
            experiment_data=experiment_data,
            group_column=group_column,
            groups=groups,
            alpha=alpha,
            power=power,
            historical_data=historical_data,
            strata=strata,
            outliers=OutlierConfig(
                handling_method=outliers_handling_method,
                threshold_quantile=outliers_threshold_quantile,
                type=outliers_type
            ),
            boot_iterations=boot_iterations,
        )

class ABTester:
    """Main class for working with A/B tests"""
    def __init__(self):
        self.builder = ABTestBuilder()

    def calculate_sample_size(
        self,
        metrics: List[str],
        effect_sizes: List[float],
        pre_experiment_data: pd.DataFrame,
        alpha: float = 0.05,
        power: float = 0.8,
        historical_data: Optional[pd.DataFrame] = None,
        strata: Optional[List[str]] = None,
        outliers_handling_method: Optional[str] = None,
        outliers_threshold_quantile: Optional[float] = None,
        outliers_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Calculate required sample size for A/B test.
        
        Parameters
        ----------
        metrics : List[str]
            List of metric names to analyze
        effect_sizes : List[float]
            List of relative changes to detect (e.g. [0.01, 0.02] for 1% and 2%)
        pre_experiment_data : pd.DataFrame
            DataFrame with pre-experiment data
        alpha : float, optional
            Significance level (Type I error rate), by default 0.05
        power : float, optional
            Statistical power (1 - Type II error rate), by default 0.8
        historical_data : pd.DataFrame, optional
            DataFrame with historical data for stratification
        strata : List[str], optional
            List of column names to use for stratification
        outliers_handling_method : str, optional
            Method for handling outliers ('remove' or 'replace_threshold')
        outliers_threshold_quantile : float, optional
            Quantile threshold for outlier detection
        outliers_type : str, optional
            Type of outliers to handle ('upper', 'lower', or 'two-sided')
        
        Returns
        -------
        pd.DataFrame
            DataFrame with required sample sizes for each metric and effect size
        """
        config = self.builder.create_sample_size_config(
            metrics=metrics,
            effect_sizes=effect_sizes,
            pre_experiment_data=pre_experiment_data,
            alpha=alpha,
            power=power,
            historical_data=historical_data,
            strata=strata,
            outliers_handling_method=outliers_handling_method,
            outliers_threshold_quantile=outliers_threshold_quantile,
            outliers_type=outliers_type,
        )
        calculator = SampleSizeCalculator(config)
        return calculator.execute()

    def run_abtest(
        self,
        metrics: List[str],
        experiment_data: pd.DataFrame,
        group_column: str = "ab_group",
        groups: Optional[List[str]] = None,
        alpha: float = 0.05,
        power: float = 0.8,
        historical_data: Optional[pd.DataFrame] = None,
        strata: Optional[List[str]] = None,
        outliers_handling_method: Optional[str] = None,
        outliers_threshold_quantile: Optional[float] = None,
        outliers_type: Optional[str] = None,
        boot_iterations: int = 10000,
    ) -> pd.DataFrame:
        """Run A/B test analysis.
        
        Parameters
        ----------
        metrics : List[str]
            List of metric names to analyze
        experiment_data : pd.DataFrame
            DataFrame with experiment data
        group_column : str, optional
            Name of column containing group labels, by default "ab_group"
        groups : List[str], optional
            List of group names [control, test]
        alpha : float, optional
            Significance level (Type I error rate), by default 0.05
        power : float, optional
            Statistical power (1 - Type II error rate), by default 0.8
        historical_data : pd.DataFrame, optional
            DataFrame with historical data for stratification
        strata : List[str], optional
            List of column names to use for stratification
        outliers_handling_method : str, optional
            Method for handling outliers ('remove' or 'replace_threshold')
        outliers_threshold_quantile : float, optional
            Quantile threshold for outlier detection
        outliers_type : str, optional
            Type of outliers to handle ('upper', 'lower', or 'two-sided')
        boot_iterations : int, optional
            Number of bootstrap iterations, by default 10000
        
        Returns
        -------
        pd.DataFrame
            DataFrame with A/B test results for each metric
        """
        config = self.builder.create_abtest_config(
            metrics=metrics,
            experiment_data=experiment_data,
            group_column=group_column,
            groups=groups,
            alpha=alpha,
            power=power,
            historical_data=historical_data,
            strata=strata,
            outliers_handling_method=outliers_handling_method,
            outliers_threshold_quantile=outliers_threshold_quantile,
            outliers_type=outliers_type,
            boot_iterations=boot_iterations,
        )
        calculator = ABTestCalculator(config)
        return calculator.execute()

def calc_evan_miller_sample_size(alpha: float, power: float, base_rate: float, pct_mde: float) -> int:
    """Calculate required sample size using Evan Miller's method.

    This function implements the sample size calculation method described by Evan Miller
    for A/B testing with binary metrics (e.g., conversion rates).

    Parameters
    ----------
    alpha : float
        Significance level (Type I error rate)
    power : float
        Statistical power (1 - Type II error rate)
    base_rate : float
        Base conversion rate in control group
    pct_mde : float
        Minimum detectable effect as percentage of base_rate

    Returns
    -------
    int
        Required sample size per group, rounded to nearest integer

    Raises
    ------
    ValueError
        If standard deviation calculation results in NaN
    """
    pct_mde = pct_mde / base_rate
    delta = base_rate * pct_mde
    t_alpha2 = ss.norm.ppf(1.0 - alpha / 2)
    t_beta = ss.norm.ppf(power)

    sd1 = np.sqrt(2 * base_rate * (1.0 - base_rate))
    sd2 = np.sqrt(base_rate * (1.0 - base_rate) + (base_rate + delta) * (1.0 - base_rate - delta))

    if np.isnan(sd1) or np.isnan(sd2):
        raise ValueError("Standard deviation is NaN. Check the input values.")
    
    return round(
        (t_alpha2 * sd1 + t_beta * sd2) * (t_alpha2 * sd1 + t_beta * sd2) / (delta * delta)
    )

def get_stratified_sample(df: pd.DataFrame, strat: str) -> pd.DataFrame:
    """Generate a stratified sample from DataFrame.

    Creates a random sample while maintaining the same proportion of samples
    for each stratum as in the original data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    strat : str
        Name of column containing stratification values

    Returns
    -------
    pd.DataFrame
        DataFrame containing stratified sample with same size as input

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'value': [1, 2, 3, 4],
    ...     'strat': ['A', 'A', 'B', 'B']
    ... })
    >>> sample = get_stratified_sample(df, 'strat')
    """
    df_size = len(df.index)
    group_size = df.shape[0]
    indices_list = []
    for group_index in set(df[strat].values):
        indices = df[df[strat] == group_index].index
        indices_list.append(np.random.choice(indices, int(group_size * len(indices) / df_size)))
    return df.loc[np.concatenate(indices_list)]

def calc_stratified_mean(
    df: pd.DataFrame, strat: str, metric: str, weights: dict
) -> float:
    """Calculate weighted stratified mean for a metric.

    Computes mean of metric for each stratum and combines them using provided weights.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    strat : str
        Name of column containing stratification values
    metric : str
        Name of column containing metric values
    weights : dict
        Dictionary mapping stratum values to their weights

    Returns
    -------
    float
        Weighted mean across all strata

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'value': [1, 2, 3, 4],
    ...     'strat': ['A', 'A', 'B', 'B']
    ... })
    >>> weights = {'A': 0.3, 'B': 0.7}
    >>> mean = calc_stratified_mean(df, 'strat', 'value', weights)
    """
    strat_mean = df.groupby(strat)[metric].mean()
    return (strat_mean * pd.Series(weights)).sum()

def calc_stratified_variance(
    df: pd.DataFrame, strat: str, metric: str, weights: dict
) -> float:
    """Calculate weighted stratified variance for a metric.

    Computes variance of metric for each stratum and combines them using provided weights.
    Missing values in groups are replaced with 0.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    strat : str
        Name of column containing stratification values
    metric : str
        Name of column containing metric values
    weights : dict
        Dictionary mapping stratum values to their weights

    Returns
    -------
    float
        Weighted variance across all strata

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'value': [1, 2, 3, 4],
    ...     'strat': ['A', 'A', 'B', 'B']
    ... })
    >>> weights = {'A': 0.3, 'B': 0.7}
    >>> var = calc_stratified_variance(df, 'strat', 'value', weights)
    """
    strat_var = df.groupby(strat)[metric].var().fillna(0)
    return (strat_var * pd.Series(weights)).sum()

def run_bootstrap_iteration(
    control: np.ndarray | pd.DataFrame,
    test: np.ndarray | pd.DataFrame,
    boot_len: int,
    is_stratified: bool,
    strat: Optional[str] = None,
    weights: Optional[dict] = None,
    metric: Optional[str] = None,
) -> Tuple[float, float, float]:
    """Execute one iteration of bootstrap resampling.

    Performs resampling with or without stratification and calculates statistics
    for both control and test groups.

    Parameters
    ----------
    control : np.ndarray | pd.DataFrame
        Control group data (DataFrame for stratified, array for simple)
    test : np.ndarray | pd.DataFrame
        Test group data (DataFrame for stratified, array for simple)
    boot_len : int
        Length of bootstrap sample
    is_stratified : bool
        Whether to use stratified sampling
    strat : str, optional
        Name of stratification column (required if is_stratified=True)
    weights : dict, optional
        Stratum weights (required if is_stratified=True)
    metric : str, optional
        Name of metric column (required if is_stratified=True)

    Returns
    -------
    Tuple[float, float, float]
        Tuple containing:
            - Control group statistic
            - Test group statistic
            - Difference (control - test)
    """
    if is_stratified:
        control_sample = get_stratified_sample(df=control, strat=strat)
        test_sample = get_stratified_sample(df=test, strat=strat)
        control_stat_value = calc_stratified_mean(df=control_sample, strat=strat, metric=metric, weights=weights)
        test_stat_value = calc_stratified_mean(df=test_sample, strat=strat, metric=metric, weights=weights)
    else:
        control_indices = np.random.randint(0, len(control), boot_len)
        test_indices = np.random.randint(0, len(test), boot_len)
        control_sample = control[control_indices]
        test_sample = test[test_indices]
        control_stat_value = np.mean(control_sample)
        test_stat_value = np.mean(test_sample)
    
    return control_stat_value, test_stat_value, control_stat_value - test_stat_value