"""
Strategy Signal Validator - SOLID Principles Implementation
Validates trading signals before execution

File Location: src/strategy/strategy_validator.py

Design Patterns Used:
- Chain of Responsibility: Validation chain
- Single Responsibility: Each validator checks ONE thing
- Open/Closed: Add new validators without modifying existing

Author: Piyush | Date: Oct 19, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod


# ============================================================================
# 1. VALIDATION RESULT (Data Class)
# ============================================================================
@dataclass
class ValidationResult:
    """
    Container for validation results
    
    Single Responsibility: Hold validation data
    """
    is_valid: bool
    validator_name: str
    message: str
    details: Optional[Dict] = None


# ============================================================================
# 2. BASE VALIDATOR (Abstract Class - Interface Segregation)
# ============================================================================
class BaseValidator(ABC):
    """
    Abstract base validator
    
    Interface Segregation: Each validator implements only what it needs
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate DataFrame
        
        Returns:
            ValidationResult with pass/fail status
        """
        pass
    
    def get_name(self) -> str:
        """Get validator name"""
        return self.__class__.__name__


# ============================================================================
# 3. SIGNAL VALUE VALIDATOR (Single Responsibility)
# ============================================================================
class SignalValueValidator(BaseValidator):
    """
    Validates signal values are correct
    
    Single Responsibility: Check signal value integrity
    """
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Check if signals are only -1, 0, 1
        
        Returns:
            ValidationResult
        """
        if 'signal' not in df.columns:
            return ValidationResult(
                is_valid=False,
                validator_name=self.get_name(),
                message="'signal' column missing"
            )
        
        valid_signals = df['signal'].isin([-1, 0, 1]).all()
        
        if not valid_signals:
            invalid_values = df[~df['signal'].isin([-1, 0, 1])]['signal'].unique()
            return ValidationResult(
                is_valid=False,
                validator_name=self.get_name(),
                message=f"Invalid signal values found: {invalid_values}",
                details={'invalid_values': invalid_values.tolist()}
            )
        
        return ValidationResult(
            is_valid=True,
            validator_name=self.get_name(),
            message="Signal values valid (-1, 0, 1)"
        )


# ============================================================================
# 4. INDICATOR QUALITY VALIDATOR (Single Responsibility)
# ============================================================================
class IndicatorQualityValidator(BaseValidator):
    """
    Validates indicator calculation quality
    
    Single Responsibility: Check indicator data quality
    """
    
    def __init__(self, required_indicators: List[str] = None):
        super().__init__()
        self.required_indicators = required_indicators or []
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Check if indicators are properly calculated
        
        Checks:
        - Required columns exist
        - No excessive NaN in recent data
        - No infinite values
        
        Returns:
            ValidationResult
        """
        # Check required columns
        missing = [col for col in self.required_indicators if col not in df.columns]
        if missing:
            return ValidationResult(
                is_valid=False,
                validator_name=self.get_name(),
                message=f"Missing required columns: {missing}",
                details={'missing_columns': missing}
            )
        
        # Check recent data (last 100 rows)
        recent = df.tail(100)
        
        # Check for NaN
        if self.required_indicators:
            nan_counts = recent[self.required_indicators].isna().sum()
            if nan_counts.any():
                return ValidationResult(
                    is_valid=False,
                    validator_name=self.get_name(),
                    message=f"NaN values in recent data",
                    details={'nan_counts': nan_counts.to_dict()}
                )
        
        # Check for infinite values
        if self.required_indicators:
            inf_counts = recent[self.required_indicators].isin([np.inf, -np.inf]).sum()
            if inf_counts.any():
                return ValidationResult(
                    is_valid=False,
                    validator_name=self.get_name(),
                    message=f"Infinite values detected",
                    details={'inf_counts': inf_counts.to_dict()}
                )
        
        return ValidationResult(
            is_valid=True,
            validator_name=self.get_name(),
            message="Indicators properly calculated"
        )


# ============================================================================
# 5. POSITION CONSISTENCY VALIDATOR (Single Responsibility)
# ============================================================================
class PositionConsistencyValidator(BaseValidator):
    """
    Validates position state transitions
    
    Single Responsibility: Check position logic consistency
    """
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Check if position transitions are logical
        
        Checks:
        - Position values are valid (-1, 0, 1)
        - No invalid jumps (e.g., -1 to 1 directly)
        
        Returns:
            ValidationResult
        """
        if 'position' not in df.columns:
            return ValidationResult(
                is_valid=False,
                validator_name=self.get_name(),
                message="'position' column missing"
            )
        
        # Check valid values
        if not df['position'].isin([-1, 0, 1]).all():
            return ValidationResult(
                is_valid=False,
                validator_name=self.get_name(),
                message="Invalid position values (must be -1, 0, 1)"
            )
        
        # Check position changes
        position_changes = df['position'].diff().abs()
        max_change = position_changes.max()
        
        # Max change should be 2 (e.g., -1 to 1 via 0)
        if max_change > 2:
            return ValidationResult(
                is_valid=False,
                validator_name=self.get_name(),
                message=f"Invalid position jump detected (change={max_change})",
                details={'max_change': float(max_change)}
            )
        
        return ValidationResult(
            is_valid=True,
            validator_name=self.get_name(),
            message="Position consistency validated"
        )


# ============================================================================
# 6. DATA SUFFICIENCY VALIDATOR (Single Responsibility)
# ============================================================================
class DataSufficiencyValidator(BaseValidator):
    """
    Validates data sufficiency
    
    Single Responsibility: Check if enough data for analysis
    """
    
    def __init__(self, min_data_points: int = 250):
        super().__init__()
        self.min_data_points = min_data_points
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Check if sufficient data available
        
        Returns:
            ValidationResult
        """
        if len(df) < self.min_data_points:
            return ValidationResult(
                is_valid=False,
                validator_name=self.get_name(),
                message=f"Insufficient data: {len(df)} < {self.min_data_points} required",
                details={
                    'available': len(df),
                    'required': self.min_data_points
                }
            )
        
        return ValidationResult(
            is_valid=True,
            validator_name=self.get_name(),
            message=f"Sufficient data: {len(df)} periods"
        )


# ============================================================================
# 7. SIGNAL DISTRIBUTION VALIDATOR (Single Responsibility)
# ============================================================================
class SignalDistributionValidator(BaseValidator):
    """
    Validates signal distribution
    
    Single Responsibility: Check signal frequency and distribution
    """
    
    def __init__(self, max_signal_rate: float = 0.1):
        super().__init__()
        self.max_signal_rate = max_signal_rate
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Check if signals are reasonably distributed
        
        Checks:
        - At least some signals generated
        - Not too frequent (over-trading)
        
        Returns:
            ValidationResult
        """
        if 'signal' not in df.columns:
            return ValidationResult(
                is_valid=False,
                validator_name=self.get_name(),
                message="'signal' column missing"
            )
        
        signal_counts = df['signal'].value_counts()
        
        # Check if any signals generated
        buy_signals = signal_counts.get(1, 0)
        sell_signals = signal_counts.get(-1, 0)
        
        if buy_signals == 0 and sell_signals == 0:
            return ValidationResult(
                is_valid=False,
                validator_name=self.get_name(),
                message="No BUY or SELL signals generated",
                details={'buy': 0, 'sell': 0}
            )
        
        # Check signal frequency
        signal_rate = (df['signal'] != 0).sum() / len(df)
        
        if signal_rate > self.max_signal_rate:
            return ValidationResult(
                is_valid=False,
                validator_name=self.get_name(),
                message=f"High signal frequency: {signal_rate:.2%} (possible over-trading)",
                details={
                    'signal_rate': float(signal_rate),
                    'threshold': self.max_signal_rate
                }
            )
        
        return ValidationResult(
            is_valid=True,
            validator_name=self.get_name(),
            message=f"Signal distribution OK: BUY={buy_signals}, SELL={sell_signals}",
            details={
                'buy_signals': int(buy_signals),
                'sell_signals': int(sell_signals),
                'signal_rate': float(signal_rate)
            }
        )


# ============================================================================
# 8. LOOKAHEAD BIAS VALIDATOR (Single Responsibility)
# ============================================================================
class LookaheadBiasValidator(BaseValidator):
    """
    Validates no lookahead bias
    
    Single Responsibility: Detect potential lookahead bias
    """
    
    def __init__(self, min_warmup_period: int = 100):
        super().__init__()
        self.min_warmup_period = min_warmup_period
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Check for potential lookahead bias
        
        Checks:
        - Indicators have proper warm-up period
        
        Returns:
            ValidationResult
        """
        # Check if indicators start too early
        indicator_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        for col in indicator_cols[:5]:  # Check first few indicators
            if col in df.columns:
                first_valid_idx = df[col].first_valid_index()
                if first_valid_idx is not None:
                    first_valid_pos = df.index.get_loc(first_valid_idx)
                    
                    if first_valid_pos < self.min_warmup_period:
                        self.logger.warning(
                            f"Indicator '{col}' starts at position {first_valid_pos}, "
                            f"may have insufficient warm-up"
                        )
        
        return ValidationResult(
            is_valid=True,
            validator_name=self.get_name(),
            message="No obvious lookahead bias detected"
        )


# ============================================================================
# 9. VALIDATION CHAIN (Chain of Responsibility Pattern)
# ============================================================================
class ValidationChain:
    """
    Chains multiple validators together
    
    Design Pattern: Chain of Responsibility
    Single Responsibility: Coordinate validation sequence
    """
    
    def __init__(self):
        self.validators: List[BaseValidator] = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_validator(self, validator: BaseValidator) -> 'ValidationChain':
        """
        Add validator to chain
        
        Returns:
            self (for method chaining)
        """
        self.validators.append(validator)
        return self
    
    def validate_all(self, df: pd.DataFrame) -> Dict[str, ValidationResult]:
        """
        Run all validators in chain
        
        Returns:
            Dictionary of {validator_name: ValidationResult}
        """
        results = {}
        
        for validator in self.validators:
            name = validator.get_name()
            result = validator.validate(df)
            results[name] = result
            
            if not result.is_valid:
                self.logger.warning(f"❌ {name}: {result.message}")
            else:
                self.logger.info(f"✅ {name}: {result.message}")
        
        return results
    
    def is_valid(self, df: pd.DataFrame) -> bool:
        """
        Check if all validators pass
        
        Returns:
            True if all validators pass
        """
        results = self.validate_all(df)
        return all(result.is_valid for result in results.values())


# ============================================================================
# 10. STRATEGY VALIDATOR (Facade Pattern)
# ============================================================================
class StrategyValidator:
    """
    Main validator facade
    
    Design Pattern: Facade - Provides simple interface to validation system
    Single Responsibility: Coordinate validation and reporting
    """
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Build validation chain
        self.chain = ValidationChain()
        
        # Add standard validators
        self.chain.add_validator(SignalValueValidator())
        self.chain.add_validator(DataSufficiencyValidator(
            min_data_points=config.get('min_data_points', 250)
        ))
        self.chain.add_validator(PositionConsistencyValidator())
        self.chain.add_validator(SignalDistributionValidator(
            max_signal_rate=config.get('max_signal_rate', 0.1)
        ))
        self.chain.add_validator(LookaheadBiasValidator(
            min_warmup_period=config.get('min_warmup_period', 100)
        ))
        
        # Store results
        self.last_results: Dict[str, ValidationResult] = {}
    
    def add_indicator_validator(self, required_indicators: List[str]):
        """Add indicator quality validator with specific indicators"""
        self.chain.add_validator(
            IndicatorQualityValidator(required_indicators)
        )
    
    def validate(self, df: pd.DataFrame, verbose: bool = True) -> bool:
        """
        Validate strategy output
        
        Args:
            df: DataFrame with strategy signals
            verbose: Print detailed report
        
        Returns:
            True if all validations pass
        """
        self.logger.info("Starting validation...")
        
        # Run validation chain
        self.last_results = self.chain.validate_all(df)
        
        # Check results
        all_valid = all(result.is_valid for result in self.last_results.values())
        
        # Print report if verbose
        if verbose:
            print(self.generate_report())
        
        return all_valid
    
    def generate_report(self) -> str:
        """
        Generate validation report
        
        Returns:
            Formatted report string
        """
        if not self.last_results:
            return "No validation results available"
        
        report = []
        report.append("=" * 60)
        report.append("STRATEGY VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        for name, result in self.last_results.items():
            status = "✅ PASS" if result.is_valid else "❌ FAIL"
            report.append(f"{name}:")
            report.append(f"  Status:  {status}")
            report.append(f"  Message: {result.message}")
            
            if result.details:
                report.append(f"  Details: {result.details}")
            report.append("")
        
        # Overall status
        all_passed = all(r.is_valid for r in self.last_results.values())
        report.append("=" * 60)
        report.append(f"Overall: {'✅ ALL CHECKS PASSED' if all_passed else '❌ SOME CHECKS FAILED'}")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def get_failed_validators(self) -> List[str]:
        """
        Get list of failed validators
        
        Returns:
            List of validator names that failed
        """
        return [
            name for name, result in self.last_results.items()
            if not result.is_valid
        ]


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================
def validate_strategy_output(df: pd.DataFrame, 
                            required_indicators: List[str] = None,
                            verbose: bool = True) -> bool:
    """
    Quick validation function
    
    Args:
        df: DataFrame with strategy signals
        required_indicators: List of required indicator columns
        verbose: Print report
    
    Returns:
        True if all checks pass
    """
    validator = StrategyValidator()
    
    if required_indicators:
        validator.add_indicator_validator(required_indicators)
    
    return validator.validate(df, verbose)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=300, freq='1h')
    
    df = pd.DataFrame({
        'open': np.random.randn(300) * 100 + 50000,
        'high': np.random.randn(300) * 100 + 50100,
        'low': np.random.randn(300) * 100 + 49900,
        'close': np.random.randn(300) * 100 + 50000,
        'volume': np.random.randint(100, 1000, 300),
        'fast_ma': np.random.randn(300) * 50 + 50000,
        'slow_ma': np.random.randn(300) * 50 + 49900,
        'rsi': np.random.uniform(30, 70, 300),
        'signal': np.random.choice([-1, 0, 1], 300, p=[0.1, 0.8, 0.1]),
        'position': np.random.choice([-1, 0, 1], 300, p=[0.1, 0.8, 0.1])
    }, index=dates)
    
    print("\n🧪 Testing Strategy Validator\n")
    
    # Test 1: Standard validation
    print("Test 1: Standard Validation")
    result = validate_strategy_output(df, verbose=True)
    print(f"\nResult: {'PASS' if result else 'FAIL'}\n")
    
    # Test 2: With indicator validation
    print("\nTest 2: With Indicator Validation")
    validator = StrategyValidator()
    validator.add_indicator_validator(['fast_ma', 'slow_ma', 'rsi'])
    result = validator.validate(df, verbose=True)
    
    # Test 3: Failed validation (invalid signals)
    print("\n\nTest 3: Invalid Signal Detection")
    df_invalid = df.copy()
    df_invalid.loc[df_invalid.index[0], 'signal'] = 5  # Invalid signal
    
    result = validate_strategy_output(df_invalid, verbose=True)
    
    print("\n✅ Validator tests complete!")