"""複数のハイパーパラメータを管理するためのクラス."""
import numpy as np
from .param import Param

class ParamList(tuple):
    """
    複数のハイパーパラメータを管理する.
    """

    def __add__(self, other):
        if isinstance(other, ParamList):
            return ParamList(tuple(self)+tuple(other))
        else:
            print(type(self), type(other))
            raise ValueError("only \'ParamList class\' is supported operand types for +")
    
    def _getValue(self):
        return np.array([param.value for param in self])
    
    def _setValue(self, new_params):
        assert len(self) == len(new_params)
        for i in range(len(new_params)):
            self[i].value = new_params[i]
    values = property(_getValue, _setValue)
    
    def _getMax(self):
        return [param.max for param in self]
    max = property(_getMax)

    def _getMin(self):
        return [param.min for param in self]
    min = property(_getMin)

    def _getName(self):
        return [param.name for param in self]
    name = property(_getName)

    def __str__(self):
        col = "No. | " + "{}".format("Parameter Name").ljust(16) + "   Value           Range                        \n"
        ret = col
        ret += "-"*len(col) + "\n"

        for No, param in enumerate(self):
            ret += f"{No:}".rjust(3) + " | " + param.__str__() + "\n"

        return ret
