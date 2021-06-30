import numpy as np
import ctypes as ct
from cgco import _cgco

# keep 4 effective digits for the fractional part if using real potentials
# make sure pairwise * smooth = unary so that the unary potentials and pairwise
# potentials are on the same scale.
_MAX_ENERGY_TERM_SCALE = 10000000 
_UNARY_FLOAT_PRECISION = 1000000
_PAIRWISE_FLOAT_PRECISION = 1000000 #1000
_SMOOTH_COST_PRECISION = 1 #100
_LABEL_COST_PRECISION = 1000000

_int_types = [np.int, np.intc, np.int32, np.int64, np.longlong]
_float_types = [np.float, np.float32, np.float64, np.float128]

_SMALL_CONSTANT = 1e-10

# error classes
class PyGcoError(Exception):
    def __init__(self, msg=''):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)

class ShapeMismatchError(PyGcoError):
    pass

class DataTypeNotSupportedError(PyGcoError):
    pass

class IndexOutOfBoundError(PyGcoError):
    pass

class IntegerOverflowRiskError(PyGcoError):
    pass


class gco(object):
    def __init__(self):
        pass

    def createGeneralGraph(self, numSites, numLabels, energyIsFloat=False):
        """Create a general graph with specified number of sites and labels. 
        If energyIsFloat is set to True, then automatic scaling and rounding
        will be applied to convert all energies to integers when running graph
        cuts. Then the final energy will be converted back to floats after the
        computation."""
        self.tempArray = np.empty(1, dtype=np.intc)
        self.energyTempArray = np.empty(1, dtype=np.longlong)
        _cgco.gcoCreateGeneralGraph(np.intc(numSites), np.intc(numLabels), self.tempArray)

        self.handle = self.tempArray[0]
        self.numSites = np.intc(numSites)
        self.numLabels = np.intc(numLabels)
        self.energyIsFloat = energyIsFloat

    def destroyGraph(self):
        _cgco.gcoDestroyGraph(self.handle)

    def _convertUnaryArray(self, e):
        if self.energyIsFloat:
            return (e * _UNARY_FLOAT_PRECISION).astype(np.intc)
        else:
            return e.astype(np.intc)

    def _convertUnaryTerm(self, e):
        if self.energyIsFloat:
            return np.intc(e * _UNARY_FLOAT_PRECISION)
        else:
            return np.intc(e)

    def _convertPairwiseArray(self, e):
        if self.energyIsFloat:
            return (e * _PAIRWISE_FLOAT_PRECISION).astype(np.intc)
        else:
            return e.astype(np.intc)

    def _convertPairwiseTerm(self, e):
        if self.energyIsFloat:
            return np.intc(e * _PAIRWISE_FLOAT_PRECISION)
        else:
            return np.intc(e)

    def _convertSmoothCostArray(self, e):
        if self.energyIsFloat:
            return (e * _SMOOTH_COST_PRECISION).astype(np.intc)
        else:
            return e.astype(np.intc)

    def _convertSmoothCostTerm(self, e):
        if self.energyIsFloat:
            return np.intc(e * _SMOOTH_COST_PRECISION)
        else:
            return np.intc(e)

    def _convertLabelCostArray(self, e):
        if self.energyIsFloat:
            return (e * _LABEL_COST_PRECISION).astype(np.intc)
        else:
            return e.astype(np.intc)

    def _convertEnergyBack(self, e):
        if self.energyIsFloat:
            return float(e) / _UNARY_FLOAT_PRECISION
        else:
            return e

    def setDataCost(self, unary):
        """Set unary potentials, unary should be a matrix of size 
        numSites x numLabels. unary can be either integers or float"""
        
        if (self.numSites, self.numLabels) != unary.shape:
            raise ShapeMismatchError(
                    "Shape of unary potentials does not match the graph.")

        # Just a reference
        self._unary = self._convertUnaryArray(unary)

        # Safety check
        if np.max(self._unary) > _MAX_ENERGY_TERM_SCALE:
            raise IntegerOverflowRiskError(
                "Unary cost maximum = {}, exceeds {}. Aborting for risk of integer overflow in gco.".format(np.max(self._unary), _MAX_ENERGY_TERM_SCALE))


        _cgco.gcoSetDataCost(self.handle, self._unary)

    def setSiteDataCost(self, site, label, e):
        """Set site data cost, dataCost(site, label) = e.
        e should be of type int or float (python primitive type)."""
        if site >= self.numSites or site < 0 or label < 0 or label >= self.numLabels:
            raise IndexOutOfBoundError()

        scaled_e = self._convertUnaryTerm(e)

        # Safety check
        if scaled_e > _MAX_ENERGY_TERM_SCALE:
            raise IntegerOverflowRiskError(
                "Unary cost = {}, exceeds {}. Aborting for risk of integer overflow in gco.".format(scaled_e, _MAX_ENERGY_TERM_SCALE))


        _cgco.gcoSetSiteDataCost(self.handle, np.intc(site), np.intc(label), scaled_e)

    def setNeighborPair(self, s1, s2, w):
        """Create an edge (s1, s2) with weight w.
        w should be of type int or float (python primitive type).
        s1 should be smaller than s2."""
        if not (0 <= s1 < s2 < self.numSites):
            raise IndexOutOfBoundError()

        scaled_w = self._convertPairwiseTerm(w)

        # Safety check
        if scaled_w > _MAX_ENERGY_TERM_SCALE:
            raise IntegerOverflowRiskError(
                "Edge weight = {}, exceeds {}. Aborting for risk of integer overflow in gco.".format(scaled_w, _MAX_ENERGY_TERM_SCALE))

        _cgco.gcoSetNeighborPair(self.handle, np.intc(s1), np.intc(s2), w)

    def setAllNeighbors(self, s1, s2, w):
        """Setup the whole neighbor system in the graph.
        s1, s2, w are 1d numpy ndarrays of the same length.

        Each element in s1 should be smaller than the corresponding element in s2.
        """
        if s1.min() < 0 or s1.max() >= self.numSites or s2.min() < 0 or s2.max() >= self.numSites:
            raise IndexOutOfBoundError()

        # These attributes are just used to keep a reference to corresponding 
        # arrays, otherwise the temporarily used arrays will be destroyed by
        # python's garbage collection system, and the C++ library won't have
        # access to them any more, which may cause trouble.
        self._edgeS1 = s1.astype(np.intc)
        self._edgeS2 = s2.astype(np.intc)
        self._edgeW = self._convertPairwiseArray(w)

        # Safety check
        if np.max(self._edgeW) > _MAX_ENERGY_TERM_SCALE:
            raise IntegerOverflowRiskError(
                "Edge weights maximum = {}, exceeds {}. Aborting for risk of integer overflow in gco.".format(np.max(self._edgeW), _MAX_ENERGY_TERM_SCALE))


        _cgco.gcoSetAllNeighbors(
                self.handle, self._edgeS1, self._edgeS2, self._edgeW, np.intc(self._edgeS1.size))

    def setSmoothCost(self, cost):
        """Set smooth cost. cost should be a symmetric numpy square matrix of 
        size numLabels x numLabels.
        
        cost[l1, l2] is the cost of labeling l1 as l2 (or l2 as l1)
        """
        if cost.shape[0] != cost.shape[1] or (cost != cost.T).any():
            raise DataTypeNotSupportedError('Cost matrix not square or not symmetric')
        if cost.shape[0] != self.numLabels:
            raise ShapeMismatchError('Cost matrix not of size numLabels * numLabels')

        # Just a reference
        self._smoothCost = self._convertSmoothCostArray(cost)

        # Safety check
        if np.max(self._smoothCost) > _MAX_ENERGY_TERM_SCALE:
            raise IntegerOverflowRiskError(
                "Smooth cost maximum = {}, exceeds {}. Aborting for risk of integer overflow in gco.".format(np.max(self._smoothCost), _MAX_ENERGY_TERM_SCALE))


        _cgco.gcoSetSmoothCost(self.handle, self._smoothCost)

    def setPairSmoothCost(self, l1, l2, cost):
        """Set smooth cost for a pair of labels l1, l2."""
        if not (0 <= l1 < self.numLabels) or not (0 <= l2 < self.numLabels):
            raise IndexOutOfBoundError()

        scaled_cost = self._convertSmoothCostTerm(cost)

        # Safety check
        if scaled_cost > _MAX_ENERGY_TERM_SCALE:
            raise IntegerOverflowRiskError(
                "Smooth cost = {}, exceeds {}. Aborting for risk of integer overflow in gco.".format(scaled_cost, _MAX_ENERGY_TERM_SCALE))


        _cgco.setPairSmoothCost(
                self.handle, np.intc(l1), np.intc(l2), scaled_cost)

    def setLabelCosts(self, costs):
        if not (len(costs) == self.numLabels):
            raise ShapeMismatchError('Label cost array does not contain {} elements'.format(self.numLabels))

        # Just a reference
        self._label_cost = self._convertLabelCostArray(costs)

        # Safety check
        if np.max(self._label_cost) > _MAX_ENERGY_TERM_SCALE:
            raise IntegerOverflowRiskError(
                "Label cost maximum = {}, exceeds {}. Aborting for risk of integer overflow in gco.".format(np.max(self._label_cost), _MAX_ENERGY_TERM_SCALE))


        _cgco.gcoSetLabelCost(
            self.handle,
            self._label_cost
        )

    def expansion(self, niters=-1):
        """Do alpha-expansion for specified number of iterations. 
        Return total energy after the expansion moves.
        If niters is set to -1, the algorithm will run until convergence."""
        _cgco.gcoExpansion(self.handle, np.intc(niters), self.energyTempArray)
        return self._convertEnergyBack(self.energyTempArray[0])
        
    def expansionOnAlpha(self, label):
        """Do one alpha-expansion move for the specified label.
        Return True if the energy decreases, return False otherwise."""
        if not (0 <= label < self.numLabels):
            raise IndexOutOfBoundError()
        _cgco.gcoExpansionOnAlpha(self.handle, np.intc(label), self.tempArray)
        return self.tempArray[0] == 1

    def swap(self, niters=-1):
        """Do alpha-beta swaps for the specified number of iterations.
        Return total energy after the swap moves.
        If niters is set to -1, the algorithm will run until convergence."""
        _cgco.gcoSwap(self.handle, np.intc(niters), self.energyTempArray)
        return self._convertEnergyBack(self.energyTempArray[0])

    def alphaBetaSwap(self, l1, l2):
        """Do a single alpha-beta swap for specified pair of labels."""
        if not (0 <= l1 < self.numLabels) or not (0 <= l2 < self.numLabels):
            raise IndexOutOfBoundError()
        _cgco.gcoAlphaBetaSwap(self.handle, np.intc(l1), np.intc(l2))

    def computeEnergy(self):
        """Compute energy of current label assignments."""
        _cgco.gcoComputeEnergy(self.handle, self.energyTempArray)
        return self._convertEnergyBack(self.energyTempArray[0])

    def computeDataEnergy(self):
        """Compute the data energy of current label assignments."""
        _cgco.gcoComputeDataEnergy(self.handle, self.energyTempArray)
        return self._convertEnergyBack(self.energyTempArray[0])

    def computeSmoothEnergy(self):
        """Compute the smooth energy of current label assignments."""
        _cgco.gcoComputeSmoothEnergy(self.handle, self.energyTempArray)
        return self._convertEnergyBack(self.energyTempArray[0])

    def getLabelAtSite(self, site):
        """Get the current label assignment at a specified site."""
        if not (0 <= site < self.numSites):
            raise IndexOutOfBoundError()
        _cgco.gcoGetLabelAtSite(self.handle, np.intc(site), self.tempArray)
        return self.tempArray[0]

    def getLabels(self):
        """Get the full label assignment for the whole graph.
        Return a 1d vector of labels of length numSites.
        """
        labels = np.empty(self.numSites, dtype=np.intc)
        _cgco.gcoGetLabels(self.handle, labels)
        return labels

    def initLabelAtSite(self, site, label):
        """Initialize label assignment at a specified site."""
        if not (0 <= site < self.numSites) or not (0 <= label < self.numLabels):
            raise IndexOutOfBoundError()
        _cgco.gcoInitLabelAtSite(self.handle, np.intc(site), np.intc(label))


def cut_general_graph(edges, edge_weights, unary_cost, pairwise_cost, label_cost=None,
        n_iter=-1, algorithm='expansion', init_labels=None, down_weight_factor=None):
    """
    Apply multi-label graph cuts to arbitrary graph given by `edges`.

    Parameters
    ----------
    edges: ndarray, int32, shape=(n_edges, 2)
        Rows correspond to edges in graph, given as vertex indices. The indices
        in the first column should always be smaller than corresponding indices
        from the second column.
    edge_weights: ndarray, int32 or float64, shape=(n_edges)
        Weights for each edge, listed in the same order as edges.
    unary_cost: ndarray, int32 or float64, shape=(n_vertices, n_labels)
        Unary potentials
    pairwise_cost: ndarray, int32 or float64, shape=(n_labels, n_labels)
        Pairwise potentials for label compatibility
    n_iter: int, (default=-1)
        Number of iterations. n_iter=-1 means run the algorithm until convergence.
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.
    init_labels: ndarray, int32, shape=(n_vertices). Initial labels.
    down_weight_factor: float or None. Used to scale down the energy terms, so
        that they won't overflow once converted to integers. Default to None,
        where this factor is set automatically.

    Return
    ------
    labels: ndarray, int32, shape=(n_vertices) the resulting list of labels
        after optimization.

    Note all the node indices start from 0.
    """
    energy_is_float = (unary_cost.dtype in _float_types) or \
            (edge_weights.dtype in _float_types) or \
            (pairwise_cost.dtype in _float_types)

    if not energy_is_float and not (
            (unary_cost.dtype in _int_types) and 
            (edge_weights.dtype in _int_types) and 
            (pairwise_cost.dtype in _int_types)):
        raise DataTypeNotSupportedError(
                "Unary and pairwise potentials should have consistent types. "
                "Either integers of floats. Mixed types or other types are not "
                "supported.")

    n_sites, n_labels = unary_cost.shape

    if down_weight_factor == None:
        down_weight_factor = max(np.abs(unary_cost).max(), 
                np.abs(edge_weights).max() * pairwise_cost.max()) + _SMALL_CONSTANT

    gc = gco()
    gc.createGeneralGraph(n_sites, n_labels, energy_is_float)
    gc.setDataCost(unary_cost / down_weight_factor)
    gc.setAllNeighbors(edges[:,0], edges[:,1], edge_weights / down_weight_factor)
    gc.setSmoothCost(pairwise_cost)

    if label_cost is not None:
        gc.setLabelCosts(label_cost / down_weight_factor)

    # initialize labels
    if init_labels is not None:
        for i in range(n_sites):
            gc.initLabelAtSite(i, init_labels[i])

    if algorithm == 'expansion':
        energy = gc.expansion(n_iter)
    else:
        energy = gc.swap(n_iter)

    labels = gc.getLabels()
    gc.destroyGraph()

    return labels, energy * down_weight_factor

def cut_grid_graph(unary_cost, pairwise_cost, costV, costH, 
        n_iter=-1, algorithm='expansion'):
    """
    Apply multi-label graphcuts to grid graph.

    Parameters
    ----------
    unary_cost: ndarray, int32, shape=(height, width, n_labels)
        Unary potentials
    pairwise_cost: ndarray, int32, shape=(n_labels, n_labels)
        Pairwise potentials for label compatibility
    costV: ndarray, int32, shape=(height-1, width)
        Vertical edge weights. 
        costV[i,j] is the edge weight between (i,j) and (i+1,j)
    costH: ndarray, int32, shape=(height, width-1)
        Horizontal edge weights.
        costH[i,j] is the edge weight between (i,j) and (i,j+1)
    n_iter: int, (default=-1)
        Number of iterations.
        Set it to -1 will run the algorithm until convergence
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.

    Note all the node indices start from 0.
    """
    energy_is_float = (unary_cost.dtype in _float_types) or \
            (pairwise_cost.dtype in _float_types) or \
            (costV.dtype in _float_types) or \
            (costH.dtype in _float_types)

    if not energy_is_float and not (
            (unary_cost.dtype in _int_types) and 
            (pairwise_cost.dtype in _int_types) and
            (costV.dtype in _int_types) and
            (costH.dtype in _int_types)):
        raise DataTypeNotSupportedError(
                "Unary and pairwise potentials should have consistent types. "
                "Either integers of floats. Mixed types or other types are not "
                "supported.")

    height, width, n_labels = unary_cost.shape

    gc = gco()
    gc.createGeneralGraph(height * width, n_labels, energy_is_float)
    gc.setDataCost(unary_cost.reshape([height * width, n_labels]))

    v_edges_from = np.arange((height-1) * width)
    v_edges_to = v_edges_from + width
    v_edges_w = costV.flatten()

    h_edges_from = np.arange(width-1)
    h_edges_from = np.tile(h_edges_from[np.newaxis, :], [height, 1])
    h_step = np.arange(height) * width
    h_edges_from = (h_edges_from + h_step[:,np.newaxis]).flatten()
    h_edges_to = h_edges_from + 1
    h_edges_w = costH.flatten()

    edges_from = np.r_[v_edges_from, h_edges_from]
    edges_to = np.r_[v_edges_to, h_edges_to]
    edges_w = np.r_[v_edges_w, h_edges_w]

    gc.setAllNeighbors(edges_from, edges_to, edges_w)

    gc.setSmoothCost(pairwise_cost)

    if algorithm == 'expansion':
        gc.expansion(n_iter)
    else:
        gc.swap(n_iter)

    labels = gc.getLabels()
    gc.destroyGraph()

    return labels

def cut_grid_graph_simple(unary_cost, pairwise_cost,
        n_iter=-1, algorithm='expansion'):
    """
    Apply multi-label graphcuts to grid graph. This is a simplified version of
    cut_grid_graph, with all edge weights set to 1.

    Parameters
    ----------
    unary_cost: ndarray, int32, shape=(height, width, n_labels)
        Unary potentials
    pairwise_cost: ndarray, int32, shape=(n_labels, n_labels)
        Pairwise potentials for label compatibility
    n_iter: int, (default=-1)
        Number of iterations.
        Set it to -1 will run the algorithm until convergence
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.

    Note all the node indices start from 0.
    """
    height, width, n_labels = unary_cost.shape
    costV = np.ones((height-1, width), dtype=unary_cost.dtype)
    costH = np.ones((height, width-1), dtype=unary_cost.dtype)

    return cut_grid_graph(unary_cost, pairwise_cost, costV, costH, n_iter, algorithm)

