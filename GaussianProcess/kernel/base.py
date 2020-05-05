"""カーネルのテンプレートと実装."""
import numpy as np
from abc import ABCMeta, abstractmethod
from ..hyper_parameter import Param, ParamList

class MyKernel(metaclass=ABCMeta):

	def __init__(self, params, name):
		self._name = name
		self._params = params
		self.n_params = len(params)

	@abstractmethod
	def getKernel(self, X1, X2, params=None, diag=False):
		"""共分散行列の取得."""
		pass

	@abstractmethod
	def paramsGrad(self):
		"""共分散行列の勾配."""
		pass

	def _getParams(self):
		return self._params

	def _setParams(self, params):
		assert len(params) == len(self._params)
		self._params.values = params
	params = property(_getParams, _setParams)

	def _getName(self):
		return self._name
	name = property(_getName)

	def __call__(self, X1, X2, params=None, diag=False):
		return self.getKernel(X1, X2, params, diag)

	def __add__(self, kern):
		"""
        Overloading + operator.

        :param kern: kernel function
        :return: an instance of SumOfKernel
        """
		return SumOfKernel(self, kern)
    
	def __mul__(self, kern):
		"""Overloading * operator."""
		if isinstance(kern, int) or isinstance(kern, float):
			return ScaleOfKernel(self, kern)
		else:
			return ProductOfKernel(self, kern)

	def __str__(self):
		ret = "Kernel function : " + self.name + "\n"
		ret += self.params.__str__()

		return ret

class SumOfKernel(MyKernel):
	"""Sum of two kernel function."""

	def __init__(self, kern1, kern2):
		self._kern1 = kern1
		self._kern2 = kern2
		super().__init__(kern1._params + kern2._params, "(" + kern1.name + "+" + kern2.name + ")")

	def getKernel(self, X1, X2, params=None, diag=False):
		if params is None:
			params = self.params.values
		l = self._kern1.n_params
		return self._kern1.getKernel(X1, X2, params[:l], diag) + self._kern2.getKernel(X1, X2, params[l:], diag)

	def paramsGrad(self, X, params=None):
		if params is None:
			params = self.params.values
		ret = np.zeros(shape=[self.n_params, X.shape[0], X.shape[0]])
		l = len(self._kern1.params)
		ret[:l] = self._kern1.paramsGrad(X, params=params[:l])
		ret[l:] = self._kern2.paramsGrad(X, params=params[l:])
		return ret

class ProductOfKernel(MyKernel):
	"""Product of two kernel function."""
	
	def __init__(self, kern1, kern2):
		self._kern1 = kern1
		self._kern2 = kern2
		super().__init__(kern1._params + kern2._params, kern1.name + "*" + kern2.name)

	def getKernel(self, X1, X2, params=None, diag=False):
		if params is None:
			params = self.params.values
		l = self._kern1.n_params
		return self._kern1.getKernel(X1, X2, params[:l], diag) * self._kern2.getKernel(X1, X2, params[l:], diag)

	def paramsGrad(self, X, params=None):
		if params is None:
			params = self.params.values
		ret = np.zeros(shape=[self.n_params, X.shape[0], X.shape[0]])
		l = len(self._kern1.params)
		ret[:l] = self._kern1.paramsGrad(X, params=params[:l]) * self._kern2.getKernel(X, X, params[l:])
		ret[l:] = self._kern2.paramsGrad(X, params=params[l:]) * self._kern1.getKernel(X, X, params[:l])
		return ret

class ScaleOfKernel(MyKernel):
	"""
	カーネルの前にスケールパラメータをつける.
	"""
	def __init__(self, kern, scalar):
		self.kern = kern
		if isinstance(kern, SumOfKernel) or isinstance(kern, ProductOfKernel):
			name = "scale*" + "(" + self.kern.name + ")"
		else:
			name = "scale*" + self.kern.name
		if kern.n_params > 0:
			params = ParamList([Param(scalar, "ln_const")]) + kern._params
		else:
			params = ParamList([Param(scalar, "ln_const")])
		super().__init__(params, name)

	def getKernel(self, X1, X2, params=None, diag=False):
		if params is None:
			params = self.params.values
		sf = np.exp(params[0])                   # scale parameter
		return sf * self.kern.getKernel(X1, X2, params[1:], diag)     # accumulate cov

	def paramsGrad(self, X, params=None):
		if params is None:
			params = self.params.values
		ret = np.zeros(shape=[self.n_params, X.shape[0], X.shape[0]])
		sf = np.exp(params[0])                 # scale parameter
		ret[0] = sf * self.kern.getKernel(X, X, params[1:])
		ret[1:] = sf * self.kern.paramsGrad(X, params=params[1:])
		return ret