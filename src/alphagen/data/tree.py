from alphagen.data.expression import *
from alphagen.data.tokens import *


class ExpressionBuilder:
    stack: List[Expression]

    def __init__(self):
        self.stack = []

    def get_tree(self) -> Expression:
        if len(self.stack) == 1:
            return self.stack[0]
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
            self.stack.append(Feature(token.feature))
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
            if not self.stack[-2].is_featured:
                return False
        elif issubclass(op, PairRollingOperator):
            if not isinstance(self.stack[-1], DeltaTime):
                return False
            if not self.stack[-2].is_featured or not self.stack[-3].is_featured:
                return False
        else:
            assert False
        return True

    def validate_dt(self) -> bool:
        return len(self.stack) > 0 and self.stack[-1].is_featured

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
            Abs, SLog1p, Inv, Sign, Log, Rank,
            Add, Sub, Mul, Div, Pow, Greater, Less,
            Ref, TsMean, TsSum, TsStd, TsIr, TsMinMaxDiff, TsMaxDiff, TsMinDiff, TsVar, TsSkew, TsKurt, TsMax, TsMin,
            TsMed, TsMad, TsRank, TsDelta, TsDiv, TsPctChange, TsWMA, TsEMA,
            TsCov, TsCorr
        )
        
        # Create operator mapping
        operator_map = {
            'Abs': Abs, 'SLog1p': SLog1p, 'Inv': Inv, 'Sign': Sign, 'Log': Log, 'Rank': Rank,
            'Add': Add, 'Sub': Sub, 'Mul': Mul, 'Div': Div, 'Pow': Pow, 'Greater': Greater, 'Less': Less,
            'Ref': Ref, 'TsMean': TsMean, 'TsSum': TsSum, 'TsStd': TsStd, 'TsIr': TsIr, 
            'TsMinMaxDiff': TsMinMaxDiff, 'TsMaxDiff': TsMaxDiff, 'TsMinDiff': TsMinDiff, 'TsVar': TsVar,
            'TsSkew': TsSkew, 'TsKurt': TsKurt, 'TsMax': TsMax, 'TsMin': TsMin, 'TsMed': TsMed, 'TsMad': TsMad,
            'TsRank': TsRank, 'TsDelta': TsDelta, 'TsDiv': TsDiv, 'TsPctChange': TsPctChange, 
            'TsWMA': TsWMA, 'TsEMA': TsEMA, 'TsCov': TsCov, 'TsCorr': TsCorr
        }
        
        # Create feature mapping
        feature_map = {
            '$open': FeatureType.OPEN,
            '$close': FeatureType.CLOSE,
            '$high': FeatureType.HIGH,
            '$low': FeatureType.LOW,
            '$volume': FeatureType.VOLUME,
            '$vwap': FeatureType.VWAP
        }
        
        tokens = []
        
        def parse_expression(expr_str):
            """Parse an expression and return tokens in RPN order"""
            expr_str = expr_str.strip()
            
            # Check if it's a feature
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
    
