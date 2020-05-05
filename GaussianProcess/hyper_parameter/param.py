"""ハイパーパラメータのテンプレートと実装."""
import numpy as np

class Param:
    """
    ハイパーパラメータのクラス.

    Attributes
    ----------
    value : float
        ハイパーパラメータの値.
    name : str
        ハイパーパラメータの名前.
    max : float
        ハイパーパラメータの上限.
    min : float
        ハイパーパラメータの下限.
    """

    def __init__(self, value=0., name=None, min=-6*np.log(10), max=6*np.log(10), log_scale=True):
        """
        コンストラクタ.

        Parameters
        ----------
        value : float, default 0.
            ハイパーパラメータの値.
        name : str or None, default None
            ハイパーパラメータの名前.
        log_scale : bool, default True
            logスケールで管理するかどうか.
        """
        self._value = value
        self._name = name
        self._max = max
        self._min = min
        self.log_scale = log_scale

    def set_value(self, value, log_scale=False):
        """
        ハイパーパラメータのセッター.

        Parameters
        ----------
        value : float
            セットする値.
        log_scale : bool, default False
            引数のvalueがlogスケールかどうか.
        """
        if log_scale:
            if self.log_scale:
                self.value = value
            else:
                self.value = np.exp(value)
        else:
            if self.log_scale:
                assert value > 0.
                self.value = np.log(value)
            else:
                self.value = value

    def set_bound(self, min_value, max_value, log_scale=False):
        """
        ハイパーパラメータの制約を設定.

        Parameters
        ----------
        max_value : float
            ハイパーパラメータの上限.
        min_value : float
            ハイパーパラメータの下限.
        log_scale : bool, default False
            引数のmin_value, max_valueがlog_scaleかどうか.
        """
        if max_value < min_value:
            print("Failed to constrain due to max_value < min_value.")
            return

        if log_scale:
            if self.log_scale:
                self._max = max_value
                self._min = min_value
            else:
                self._max = np.exp(max_value)
                self._min = np.exp(min_value)
        else:
            if self.log_scale:
                assert max_value > 0. and min_value > 0.
                self._max = np.log(max_value)
                self._min = np.log(min_value)
            else:
                self._max = max_value
                self._min = min_value

        self.value = self._value

    def _getName(self):
        return self._name
    name = property(_getName)

    def _getMin(self):
        return self._min
    min = property(_getMin)

    def _getMax(self):
        return self._max
    max = property(_getMax)

    def _getValue(self):
        return self._value
    
    def _setValue(self, value):
        self._value = np.minimum(np.maximum(self._min, value), self._max)
    value = property(_getValue, _setValue)

    def __str__(self):
        if self.log_scale:
            name = self.name[3:]
            val = np.exp(self.value)
            max = np.exp(self.max)
            min = np.exp(self.min)
        else:
            name = self.name
            val = self.value
            max = self.max
            min = self.min

        ret = f"{name:}".ljust(16)+"   " + f"{val:6.6f}".rjust(13)+ \
            "   ["+f"{min:6.6f}".rjust(13)+", "+f"{max:6.6f}".rjust(13)+"]"

        return ret