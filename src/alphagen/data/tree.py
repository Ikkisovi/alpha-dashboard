from alphagen.data.expression import *
from alphagen.data.tokens import *



def is_minute_level(expr: Expression) -> bool:
    """
    Check if an expression is strictly minute-level (contains no daily aggregation).
    """
    return expression_output_layer(expr) == "minute"


class ExpressionBuilder:
    stack: List[Expression]

    def __init__(self):
        self.stack = []

    # ------------------------------------------------------------------ #
    # Helpers for structural validation
    # ------------------------------------------------------------------ #
    def _expr_children(self, expr: Expression) -> List[Expression]:
        if isinstance(expr, UnaryOperator):
            return [expr._operand]
        if isinstance(expr, BinaryOperator):
            return [expr._lhs, expr._rhs]
        if isinstance(expr, RollingOperator):
            children = [expr._operand]
            # TsQuantile carries an additional quantile expression we want to traverse
            quantile = getattr(expr, "_quantile", None)
            if isinstance(quantile, Expression):
                children.append(quantile)
            return children
        if isinstance(expr, PairRollingOperator):
            return [expr._lhs, expr._rhs]
        if isinstance(expr, IntraRef):
            return [expr._operand]
        if isinstance(expr, IntradayAggregator):
            return [expr._operand]
        if isinstance(expr, IntraSumRatio):
            return [expr._lhs, expr._rhs]
        return []

    def _contains_sign(self, expr: Expression) -> bool:
        if isinstance(expr, Sign):
            return True
        return any(self._contains_sign(child) for child in self._expr_children(expr) if isinstance(child, Expression))

    def get_tree(self) -> Expression:
        if len(self.stack) == 1:
            expr = self.stack[0]
            # Disallow minute-lag operators unless they are nested inside an intraday aggregator
            self._validate_intra_refs(expr, in_intraday=False)
            return expr
        else:
            raise InvalidExpressionException(f"Expected only one tree, got {len(self.stack)}")
    
    def get_ast(self):
        pass

    def add_token(self, token: Token):
        if not self.validate(token):
            raise InvalidExpressionException(f"Token {token} not allowed here, stack: {self.stack}.")
        if isinstance(token, OperatorToken):
            n_args: int = token.operator.n_args()
            children = []
            for _ in range(n_args):
                children.append(self.stack.pop())
            self.stack.append(token.operator(*reversed(children)))  # type: ignore
        elif isinstance(token, ConstantToken):
            self.stack.append(Constant(token.constant))
        elif isinstance(token, DeltaTimeToken):
            self.stack.append(DeltaTime(token.delta_time))
        elif isinstance(token, FeatureToken):
            self.stack.append(Feature(token.feature, getattr(token, "is_minute", False)))
        else:
            assert False

    def is_valid(self) -> bool:
        return len(self.stack) == 1 and self.stack[0].is_featured

    def validate(self, token: Token) -> bool:
        if isinstance(token, OperatorToken):
            return self.validate_op(token.operator)
        elif isinstance(token, DeltaTimeToken):
            return self.validate_dt()
        elif isinstance(token, ConstantToken):
            return self.validate_const()
        elif isinstance(token, FeatureToken):
            return self.validate_feature()
        elif isinstance(token, SequenceIndicatorToken):
            return True
        else:
            assert False

    def validate_op(self, op: Type[Operator]) -> bool:
        if len(self.stack) < op.n_args():
            return False


        if issubclass(op, UnaryOperator):
            if not self.stack[-1].is_featured:
                return False
        elif issubclass(op, BinaryOperator):
            if not self.stack[-1].is_featured and not self.stack[-2].is_featured:
                return False
            if (isinstance(self.stack[-1], DeltaTime) or
                    isinstance(self.stack[-2], DeltaTime)):
                return False
        elif issubclass(op, RollingOperator):
            if not isinstance(self.stack[-1], DeltaTime):
                return False
            if op.n_args() == 2:
                if not self.stack[-2].is_featured:
                    return False
            elif op.n_args() == 3:
                if isinstance(self.stack[-2], DeltaTime) and op != TsRelStrength:
                    return False
                # TsRelStrength requires constant for middle arg and 0 < fast < slow
                if op == TsRelStrength:
                    if not isinstance(self.stack[-2], DeltaTime):
                        return False
                    fast = self.stack[-2]._delta_time
                    slow = self.stack[-1]._delta_time
                    # fast_window must be > 0 and < slow_window
                    if fast <= 0 or fast >= slow:
                        return False
                if not self.stack[-3].is_featured:
                    return False
            else:
                return False
        elif issubclass(op, PairRollingOperator):
            if not isinstance(self.stack[-1], DeltaTime):
                return False
            if not self.stack[-2].is_featured or not self.stack[-3].is_featured:
                return False
        elif issubclass(op, IntraRef):
            # Minute-level lag: operand then minute offset
            if not isinstance(self.stack[-1], DeltaTime):
                return False
            if not self.stack[-2].is_featured:
                return False
            if expression_output_layer(self.stack[-2]) != "minute":
                return False
        elif issubclass(op, IntradayAggregator):
            # IntradayAggregator takes one featured operand (no DeltaTime)
            if not self.stack[-1].is_featured:
                return False
            if isinstance(self.stack[-1], DeltaTime):
                return False
            # Must be minute-level input
            if expression_output_layer(self.stack[-1]) != "minute":
                return False
        elif issubclass(op, IntraSumRatio):
            # IntraSumRatio takes two featured operands (no DeltaTime)
            if not self.stack[-1].is_featured or not self.stack[-2].is_featured:
                return False
            if isinstance(self.stack[-1], DeltaTime) or isinstance(self.stack[-2], DeltaTime):
                return False
            # Must be minute-level input
            if expression_output_layer(self.stack[-1]) != "minute" or expression_output_layer(self.stack[-2]) != "minute":
                return False
        else:
            assert False
        return True

    def _validate_intra_refs(self, node: Expression, in_intraday: bool):
        """Ensure IntraRef only appears inside intraday aggregators."""
        from alphagen.data.expression import IntraRef, IntradayAggregator
        if isinstance(node, IntraRef) and not in_intraday:
            raise InvalidExpressionException("IntraRef must be used inside an intraday aggregator.")
        if isinstance(node, IntradayAggregator):
            child_in_intraday = True
        else:
            child_in_intraday = in_intraday

        children = []
        if isinstance(node, IntraRef):
            children = [node._operand]
        elif isinstance(node, UnaryOperator):
            children = [node._operand]
        elif isinstance(node, BinaryOperator):
            children = [node._lhs, node._rhs]
        elif isinstance(node, RollingOperator):
            children = [node._operand]
        elif isinstance(node, PairRollingOperator):
            children = [node._lhs, node._rhs]
        if isinstance(node, IntradayAggregator):
            children = [node._operand]

        for child in children:
            if isinstance(child, Expression):
                self._validate_intra_refs(child, child_in_intraday)

    def validate_dt(self) -> bool:
        if len(self.stack) == 0:
            return False
        if self.stack[-1].is_featured:
            return True
        return len(self.stack) >= 2 and not self.stack[-1].is_featured and self.stack[-2].is_featured

    def validate_const(self) -> bool:
        return len(self.stack) == 0 or self.stack[-1].is_featured

    def validate_feature(self) -> bool:
        return not (len(self.stack) >= 1 and isinstance(self.stack[-1], DeltaTime))

class ExpressionParser:
    def __init__(self):
        self.stack = []
        self.tokens = []

    def tokenize(self, expr: str) -> List[Token]:
        from alphagen.data.expression import (
            Abs, SLog1p, Inv, Sign, Log, Rank, Sqrt, Ret,
            Add, Sub, Mul, Div, Pow, Greater, Less, Corr, Quantile,
            Ref, TsMean, TsSum, TsStd, TsIr, TsMinMaxDiff, TsMaxDiff, TsMinDiff, TsVar, TsSkew, TsKurt, TsMax, TsMin,
            TsMed, TsMad, TsRank, TsDelta, TsRet, TsDiv, TsArgMax, TsArgMin, TsPctChange, TsWMA, TsEMA, TsQuantile,
            TsSortino, TsMomRank, TsMaxDd, TsRelStrength,
            TsCov, TsCorr,
            IntraMean, IntraStd, IntraLast, IntraFirst, IntraMax, IntraMin, IntraRange,
            IntraSum, IntraMedian, IntraVar, IntraSkew, IntraKurt, IntraSumRatio, IntraRef,
            Gt, Ge, Lt, Le, Having, NotHaving
        )
        
        # Create operator mapping
        operator_map = {
            'Abs': Abs, 'SLog1p': SLog1p, 'Inv': Inv, 'Sign': Sign, 'Log': Log, 'Rank': Rank, 'Sqrt': Sqrt, 'Ret': Ret,
            'Add': Add, 'Sub': Sub, 'Mul': Mul, 'Div': Div, 'Pow': Pow, 'Greater': Greater, 'Less': Less, 'Corr': Corr, 'Quantile': Quantile,
            'Ref': Ref, 'TsMean': TsMean, 'TsSum': TsSum, 'TsStd': TsStd, 'TsIr': TsIr, 
            'TsMinMaxDiff': TsMinMaxDiff, 'TsMaxDiff': TsMaxDiff, 'TsMinDiff': TsMinDiff, 'TsVar': TsVar,
            'TsSkew': TsSkew, 'TsKurt': TsKurt, 'TsMax': TsMax, 'TsMin': TsMin, 'TsMed': TsMed, 'TsMad': TsMad,
            'TsRank': TsRank, 'TsDelta': TsDelta, 'TsRet': TsRet, 'TsDiv': TsDiv, 'TsArgMax': TsArgMax, 'TsArgMin': TsArgMin,
            'TsPctChange': TsPctChange, 'TsWMA': TsWMA, 'TsEMA': TsEMA, 'TsQuantile': TsQuantile,
            'TsSortino': TsSortino, 'TsMomRank': TsMomRank, 'TsMaxDd': TsMaxDd, 'TsRelStrength': TsRelStrength,
            'TsCov': TsCov, 'TsCorr': TsCorr,
            # Intraday
            'IntraMean': IntraMean, 'IntraStd': IntraStd, 'IntraLast': IntraLast, 
            'IntraFirst': IntraFirst, 'IntraMax': IntraMax, 'IntraMin': IntraMin, 
            'IntraRange': IntraRange, 'IntraSum': IntraSum, 'IntraMedian': IntraMedian, 
            'IntraVar': IntraVar, 'IntraSkew': IntraSkew, 'IntraKurt': IntraKurt,
            'IntraSumRatio': IntraSumRatio, 'IntraRef': IntraRef,
            # Transformational
            'Gt': Gt, 'Ge': Ge, 'Lt': Lt, 'Le': Le, 'Having': Having, 'NotHaving': NotHaving
        }
        
        # Create feature mapping
        feature_map = {
            '$open': FeatureType.OPEN,
            '$close': FeatureType.CLOSE,
            '$high': FeatureType.HIGH,
            '$low': FeatureType.LOW,
            '$volume': FeatureType.VOLUME,
            '$vwap': FeatureType.VWAP,
            '$log_volume': FeatureType.LOG_VOLUME,
            '$log_money': FeatureType.LOG_MONEY,
            '$log_close': FeatureType.LOG_CLOSE,
            # Derived factors exposed directly
            '$drift_factor': FeatureType.DRIFT_FACTOR,
            '$amihud_mean': FeatureType.AMIHUD_MEAN,
            '$amihud_range': FeatureType.AMIHUD_RANGE,
            # Minute features
            '$m_open': FeatureType.OPEN,
            '$m_close': FeatureType.CLOSE,
            '$m_high': FeatureType.HIGH,
            '$m_low': FeatureType.LOW,
            '$m_volume': FeatureType.VOLUME,
            '$m_vwap': FeatureType.VWAP,
            '$m_return': FeatureType.CLOSE, # Proxy for return
        }
        minute_feature_keys = {k for k in feature_map if k.startswith('$m_')}
        
        tokens = []
        
        def parse_expression(expr_str):
            """Parse an expression and return tokens in RPN order"""
            expr_str = expr_str.strip()
            
            # Check if it's a feature (minute-level support removed; treat all as daily)
            if expr_str in feature_map:
                tokens.append(FeatureToken(feature_map[expr_str]))
                return
            
            # Check if it's a number
            if expr_str.replace('-', '').replace('.', '').isdigit() or (expr_str.startswith('-') and expr_str[1:].replace('.', '').isdigit()):
                num = float(expr_str)
                # Check if the original string contains a decimal point to distinguish float from int
                if '.' in expr_str:
                    # It's a float, so it's a constant
                    tokens.append(ConstantToken(num))
                else:
                    # It's an integer, so it's a delta time
                    tokens.append(DeltaTimeToken(int(num)))
                return
            
            # Check if it's a function call
            if '(' in expr_str and expr_str.endswith(')'):
                # Extract function name and arguments
                func_name = expr_str[:expr_str.find('(')]
                args_str = expr_str[expr_str.find('(')+1:expr_str.rfind(')')]
                
                if func_name not in operator_map:
                    raise ValueError(f"Unknown operator: {func_name}")
                
                # Parse arguments
                args = split_arguments(args_str)

                # Special-case TsQuantile so that legacy 2-arg usage (operand, window)
                # is still accepted by injecting a median quantile of 0.5.
                if func_name == 'TsQuantile' and len(args) == 2:
                    parse_expression(args[0])
                    tokens.append(ConstantToken(0.5))
                    parse_expression(args[1])
                    tokens.append(OperatorToken(operator_map[func_name]))
                    return
                
                # Recursively parse each argument (this produces operands first)
                for arg in args:
                    parse_expression(arg)
                
                # Add the operator token (this comes after operands in RPN)
                tokens.append(OperatorToken(operator_map[func_name]))
                return
            
            raise ValueError(f"Invalid expression: {expr_str}")
        
        def split_arguments(args_str):
            """Split comma-separated arguments, handling nested parentheses"""
            args = []
            current_arg = ""
            paren_count = 0
            
            for char in args_str:
                if char == '(':
                    paren_count += 1
                    current_arg += char
                elif char == ')':
                    paren_count -= 1
                    current_arg += char
                elif char == ',' and paren_count == 0:
                    args.append(current_arg.strip())
                    current_arg = ""
                else:
                    current_arg += char
            
            if current_arg.strip():
                args.append(current_arg.strip())
            
            return args
        
        parse_expression(expr)
        return tokens

    def parse(self, expr: str) -> Expression:
        self.tokens = self.tokenize(expr)
        # Create a new builder for each parse operation
        builder = ExpressionBuilder()
        for token in self.tokens:
            builder.add_token(token)
        return builder.get_tree()


class InvalidExpressionException(ValueError):
    pass


if __name__ == '__main__':
    # Test 1: Original example
    print("=== Test 1: Original example ===")
    tokens = [
        FeatureToken(FeatureType.LOW),
        OperatorToken(Abs),
        DeltaTimeToken(-10),
        OperatorToken(Ref),
        FeatureToken(FeatureType.HIGH),
        FeatureToken(FeatureType.CLOSE),
        OperatorToken(Div),
        OperatorToken(Add),
    ]

    builder = ExpressionBuilder()
    for token in tokens:
        builder.add_token(token)

    print(f'res: {str(builder.get_tree())}')
    print(f'ref: Add(Ref(Abs($low),-10),Div($high,$close))')
    
    # Test with parser
    ast_parser = ExpressionParser()
    parsed_expr = ast_parser.parse("Add(Ref(Abs($low),-10),Div($high,$close))")
    print(f'parsed: {str(parsed_expr)}')
    print(f'match: {str(builder.get_tree()) == str(parsed_expr)}')
    print()

    # Test 2: Simple unary operator
    print("=== Test 2: Simple unary operator ===")
    test_expr = "Abs($high)"
    parsed = ast_parser.parse(test_expr)
    print(f'input: {test_expr}')
    print(f'output: {str(parsed)}')
    print()

    # Test 3: Simple binary operator
    print("=== Test 3: Simple binary operator ===")
    test_expr = "Add($open,$close)"
    parsed = ast_parser.parse(test_expr)
    print(f'input: {test_expr}')
    print(f'output: {str(parsed)}')
    print()

    # Test 4: Rolling operator with time window
    print("=== Test 4: Rolling operator with time window ===")
    test_expr = "TsMean($volume,5)"
    parsed = ast_parser.parse(test_expr)
    print(f'input: {test_expr}')
    print(f'output: {str(parsed)}')
    print()

    # Test 5: Complex nested expression
    print("=== Test 5: Complex nested expression ===")
    test_expr = "Div(Sub($high,$low),Add($open,$close))"
    parsed = ast_parser.parse(test_expr)
    print(f'input: {test_expr}')
    print(f'output: {str(parsed)}')
    print()

    # Test 6: Multiple rolling operators
    print("=== Test 6: Multiple rolling operators ===")
    test_expr = "TsStd(TsMean($volume,10),5)"
    parsed = ast_parser.parse(test_expr)
    print(f'input: {test_expr}')
    print(f'output: {str(parsed)}')
    print()

    # Test 7: Pair rolling operator
    print("=== Test 7: Pair rolling operator ===")
    test_expr = "TsCorr($high,$low,20)"
    parsed = ast_parser.parse(test_expr)
    print(f'input: {test_expr}')
    print(f'output: {str(parsed)}')
    print()

    # Test 8: Expression with constants
    print("=== Test 8: Expression with constants ===")
    test_expr = "Add($close,1.5)"
    parsed = ast_parser.parse(test_expr)
    print(f'input: {test_expr}')
    print(f'output: {str(parsed)}')
    print()

    # Test 9: Complex expression with multiple operators
    print("=== Test 9: Complex expression with multiple operators ===")
    test_expr = "Mul(Add($open,$close),Div(Sub($high,$low),2.0))"
    parsed = ast_parser.parse(test_expr)
    print(f'input: {test_expr}')
    print(f'output: {str(parsed)}')
    print()

    # Test 10: All feature types
    print("=== Test 10: All feature types ===")
    features = ["$open", "$close", "$high", "$low", "$volume", "$vwap"]
    for feature in features:
        test_expr = f"Abs({feature})"
        parsed = ast_parser.parse(test_expr)
        print(f'input: {test_expr}')
        print(f'output: {str(parsed)}')
    print()

    # Test 11: Various unary operators
    print("=== Test 11: Various unary operators ===")
    unary_ops = ["Abs", "Log", "Sign", "Rank", "Inv", "SLog1p"]
    for op in unary_ops:
        test_expr = f"{op}($close)"
        parsed = ast_parser.parse(test_expr)
        print(f'input: {test_expr}')
        print(f'output: {str(parsed)}')
    print()

    # Test 12: Various binary operators
    print("=== Test 12: Various binary operators ===")
    binary_ops = ["Add", "Sub", "Mul", "Div", "Pow", "Greater", "Less"]
    for op in binary_ops:
        test_expr = f"{op}($open,$close)"
        parsed = ast_parser.parse(test_expr)
        print(f'input: {test_expr}')
        print(f'output: {str(parsed)}')
    print()

    # Test 13: Various rolling operators
    print("=== Test 13: Various rolling operators ===")
    rolling_ops = ["TsMean", "TsSum", "TsStd", "TsMax", "TsMin", "TsVar", "TsSkew", "TsKurt"]
    for op in rolling_ops:
        test_expr = f"{op}($volume,10)"
        parsed = ast_parser.parse(test_expr)
        print(f'input: {test_expr}')
        print(f'output: {str(parsed)}')
    print()
    
    # Test 14: Edge case - constant first
    print("=== Test 14: Edge case - constant first ===")
    test_expr = "Div(Greater(30.0,$low),$high)"
    parsed = ast_parser.parse(test_expr)
    print(f'input: {test_expr}')
    print(f'output: {str(parsed)}')
    print()

    # Test 15: Edge case - single feature
    print("=== Test 15: Edge case - single feature ===")
    test_expr = "$close"
    parsed = ast_parser.parse(test_expr)
    print(f'input: {test_expr}')
    print(f'output: {str(parsed)}')
    print()

    # Test 16: Edge case - single constant
    print("=== Test 16: Edge case - single constant ===")
    test_expr = "1.5"
    parsed = ast_parser.parse(test_expr)
    print(f'input: {test_expr}')
    print(f'output: {str(parsed)}')
    print()

    # Test 17: Very complex nested expression
    print("=== Test 17: Very complex nested expression ===")
    test_expr = "Add(Mul(Ref($close,-1),TsMean($volume,5)),Div(Sub($high,$low),TsStd($open,10)))"
    parsed = ast_parser.parse(test_expr)
    print(f'input: {test_expr}')
    print(f'output: {str(parsed)}')
    print()

    # Test 18: Test tokenization separately
    print("=== Test 18: Test tokenization separately ===")
    test_expr = "Add(Ref(Abs($low),-10),Div($high,$close))"
    tokens = ast_parser.tokenize(test_expr)
    print(f'input: {test_expr}')
    print(f'tokens: {[str(t) for t in tokens]}')
    print()

    # Test 19: Error handling - invalid operator
    print("=== Test 19: Error handling - invalid operator ===")
    try:
        test_expr = "InvalidOp($close)"
        parsed = ast_parser.parse(test_expr)
        print(f'input: {test_expr}')
        print(f'output: {str(parsed)}')
    except Exception as e:
        print(f'input: {test_expr}')
        print(f'error: {e}')
    print()

    # Test 20: Error handling - invalid feature
    print("=== Test 20: Error handling - invalid feature ===")
    try:
        test_expr = "Abs($invalid)"
        parsed = ast_parser.parse(test_expr)
        print(f'input: {test_expr}')
        print(f'error: {e}')
    except Exception as e:
        print(f'input: {test_expr}')
        print(f'error: {e}')
    print()

    print("=== All tests completed ===")
    
