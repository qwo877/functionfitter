import torch
import numpy as np
import random
import ast
import operator

class FunctionGen:
    def __init__(self, inF=None):
        self.inF = inF
        
        if self.inF is None:
            self.freq_low = random.uniform(0.5, 1.5)
            self.amp_low = random.uniform(1.0, 2.0)
            self.freq_high = random.uniform(3.0, 6.0) 
            self.amp_high = random.uniform(0.5, 1.0)
            self.phase = random.uniform(0, np.pi)

        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,  
        }

        self.allowed_funcs = {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
            'abs': np.abs, 'power': np.power, 'tanh': np.tanh
        }

    def _safe_eval(self, node, x_val):
        if isinstance(node, ast.Constant): 
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Name):
            if node.id == 'x':
                return x_val
            elif node.id == 'pi':
                return np.pi
            elif node.id == 'e':
                return np.e
            else:
                raise ValueError(f"不允許的變數: {node.id}")

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in self.operators:
                return self.operators[op_type](
                    self._safe_eval(node.left, x_val),
                    self._safe_eval(node.right, x_val)
                )
            else:
                raise ValueError(f"不支援的運算符號: {op_type}")

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in self.operators:
                return self.operators[op_type](self._safe_eval(node.operand, x_val))
            else:
                raise ValueError(f"不支援的一元運算: {op_type}")

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in self.allowed_funcs:
                args = [self._safe_eval(arg, x_val) for arg in node.args]
                return self.allowed_funcs[node.func.id](*args)
            else:
                func_name = node.func.id if isinstance(node.func, ast.Name) else str(node.func)
                raise ValueError(f"不允許或未知的函數: {func_name}")

        else:
            raise ValueError(f"不支援的語法結構: {type(node)}")

    def true_function(self, x):
        if self.inF:
            try:
                tree = ast.parse(self.inF, mode='eval')
                y = self._safe_eval(tree.body, x)
                
                if isinstance(y, (int, float)):
                    y = np.full_like(x, y)
                return y
            except Exception as e:
                raise ValueError(f"解析函數 '{self.inF}' 失敗: {e}")

        else:
            term1 = self.amp_low * np.sin(self.freq_low * x + self.phase)
            term2 = self.amp_high * np.cos(self.freq_high * x)
            bias = 0.1 * (x ** 2)
            return term1 + term2 - bias

    def get_data(self, num_points=100, x_range=(-4, 4)):
        x_dense = np.linspace(x_range[0], x_range[1], 300)
        y_dense = self.true_function(x_dense)
        
        x_train = np.random.uniform(x_range[0], x_range[1], num_points)
        y_train = self.true_function(x_train)
        
        x_train_tensor = torch.FloatTensor(x_train).unsqueeze(1)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        return x_dense, y_dense, x_train_tensor, y_train_tensor