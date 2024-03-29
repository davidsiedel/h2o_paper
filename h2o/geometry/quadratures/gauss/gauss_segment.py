from h2o.geometry.quadratures.quadrature import *


def get_number_of_quadrature_points_in_segment(integration_order: int) -> int:
    """

    Args:
        integration_order:

    Returns:

    """
    qw = get_reference_segment_quadrature_item(integration_order, QuadratureItem.WEIGHTS)
    number_of_quadrature_points_in_segment = len(qw)
    return number_of_quadrature_points_in_segment


def get_reference_segment_quadrature_item(integration_order: int, quadrature_item: QuadratureItem) -> ndarray:
    """

    Args:
        integration_order:

    Returns:

    """
    if integration_order == 0:
        reference_points = [
            [0.5000000000000000, 0.5000000000000000],
        ]
        reference_weights = [
            1.0000000000000000,
        ]
        jacobian = [
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
        ]
    elif integration_order == 1:
        reference_points = [
            [0.5000000000000000, 0.5000000000000000],
        ]
        reference_weights = [
            1.0000000000000000,
        ]
        jacobian = [
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
        ]
    elif integration_order == 2:
        reference_points = [
            [0.7886751345948129, 0.2113248654051871],
            [0.2113248654051871, 0.7886751345948129],
        ]
        reference_weights = [
            0.5000000000000000,
            0.5000000000000000,
        ]
        jacobian = [
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
        ]
    elif integration_order == 3:
        reference_points = [
            [0.8872983346207417, 0.1127016653792583],
            [0.5000000000000000, 0.5000000000000000],
            [0.1127016653792583, 0.8872983346207417],
        ]
        reference_weights = [
            0.2777777777777777,
            0.4444444444444446,
            0.2777777777777777,
        ]
        jacobian = [
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
        ]
    elif integration_order == 4:
        reference_points = [
            [0.9305681557970262, 0.0694318442029737],
            [0.6699905217924281, 0.3300094782075719],
            [0.3300094782075719, 0.6699905217924281],
            [0.0694318442029737, 0.9305681557970262],
        ]
        reference_weights = [
            0.1739274225687268,
            0.3260725774312733,
            0.3260725774312733,
            0.1739274225687268,
        ]
        jacobian = [
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
        ]
    elif integration_order == 5:
        reference_points = [
            [0.9530899229693320, 0.0469100770306680],
            [0.7692346550528415, 0.2307653449471584],
            [0.5000000000000000, 0.5000000000000000],
            [0.2307653449471584, 0.7692346550528415],
            [0.0469100770306680, 0.9530899229693320],
        ]
        reference_weights = [
            0.1184634425280945,
            0.2393143352496831,
            0.2844444444444445,
            0.2393143352496831,
            0.1184634425280945,
        ]
        jacobian = [
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
        ]
    elif integration_order == 6:
        reference_points = [
            [0.9662347571015760, 0.0337652428984240],
            [0.8306046932331322, 0.1693953067668678],
            [0.6193095930415985, 0.3806904069584016],
            [0.3806904069584016, 0.6193095930415985],
            [0.1693953067668678, 0.8306046932331322],
            [0.0337652428984240, 0.9662347571015760],
        ]
        reference_weights = [
            0.0856622461895852,
            0.1803807865240694,
            0.2339569672863455,
            0.2339569672863455,
            0.1803807865240694,
            0.0856622461895852,
        ]
        jacobian = [
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
        ]
    elif integration_order == 7:
        reference_points = [
            [0.9745539561713792, 0.0254460438286208],
            [0.8707655927996972, 0.1292344072003028],
            [0.7029225756886985, 0.2970774243113014],
            [0.5000000000000000, 0.5000000000000000],
            [0.2970774243113014, 0.7029225756886985],
            [0.1292344072003028, 0.8707655927996972],
            [0.0254460438286208, 0.9745539561713792],
        ]
        reference_weights = [
            0.0647424830844351,
            0.1398526957446383,
            0.1909150252525594,
            0.2089795918367346,
            0.1909150252525594,
            0.1398526957446383,
            0.0647424830844351,
        ]
        jacobian = [
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
        ]
    elif integration_order == 8:
        reference_points = [
            [0.9801449282487681, 0.0198550717512319],
            [0.8983332387068134, 0.1016667612931866],
            [0.7627662049581645, 0.2372337950418355],
            [0.5917173212478249, 0.4082826787521751],
            [0.4082826787521751, 0.5917173212478249],
            [0.2372337950418355, 0.7627662049581645],
            [0.1016667612931866, 0.8983332387068134],
            [0.0198550717512319, 0.9801449282487681],
        ]
        reference_weights = [
            0.0506142681451882,
            0.1111905172266871,
            0.1568533229389437,
            0.1813418916891810,
            0.1813418916891810,
            0.1568533229389437,
            0.1111905172266871,
            0.0506142681451882,
        ]
        jacobian = [
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
            [
                [-1.0000000000000000, 1.0000000000000000],
            ],
        ]
    # if integration_order in [0, 1]:
    #     reference_points = [
    #         [0.5000000000000000, 0.5000000000000000],
    #     ]
    #     reference_weights = [
    #         1.0000000000000000,
    #     ]
    # elif integration_order == 2:
    #     reference_points = [
    #         [0.7886751345948129, 0.2113248654051871],
    #         [0.2113248654051871, 0.7886751345948129],
    #     ]
    #     reference_weights = [
    #         0.5000000000000000,
    #         0.5000000000000000,
    #     ]
    # elif integration_order == 3:
    #     reference_points = [
    #         [0.8872983346207417, 0.1127016653792583],
    #         [0.5000000000000000, 0.5000000000000000],
    #         [0.1127016653792583, 0.8872983346207417],
    #     ]
    #     reference_weights = [
    #         0.2777777777777777,
    #         0.4444444444444446,
    #         0.2777777777777777,
    #     ]
    # elif integration_order == 4:
    #     reference_points = [
    #         [0.9305681557970262, 0.0694318442029737],
    #         [0.6699905217924281, 0.3300094782075719],
    #         [0.3300094782075719, 0.6699905217924281],
    #         [0.0694318442029737, 0.9305681557970262],
    #     ]
    #     reference_weights = [
    #         0.1739274225687268,
    #         0.3260725774312733,
    #         0.3260725774312733,
    #         0.1739274225687268,
    #     ]
    # elif integration_order == 5:
    #     reference_points = [
    #         [0.9530899229693320, 0.0469100770306680],
    #         [0.7692346550528415, 0.2307653449471584],
    #         [0.5000000000000000, 0.5000000000000000],
    #         [0.2307653449471584, 0.7692346550528415],
    #         [0.0469100770306680, 0.9530899229693320],
    #     ]
    #     reference_weights = [
    #         0.1184634425280945,
    #         0.2393143352496831,
    #         0.2844444444444445,
    #         0.2393143352496831,
    #         0.1184634425280945,
    #     ]
    # elif integration_order == 6:
    #     reference_points = [
    #         [0.9662347571015760, 0.0337652428984240],
    #         [0.8306046932331322, 0.1693953067668678],
    #         [0.6193095930415985, 0.3806904069584016],
    #         [0.3806904069584016, 0.6193095930415985],
    #         [0.1693953067668678, 0.8306046932331322],
    #         [0.0337652428984240, 0.9662347571015760],
    #     ]
    #     reference_weights = [
    #         0.0856622461895852,
    #         0.1803807865240694,
    #         0.2339569672863455,
    #         0.2339569672863455,
    #         0.1803807865240694,
    #         0.0856622461895852,
    #     ]
    # elif integration_order == 7:
    #     reference_points = [
    #         [0.9745539561713792, 0.0254460438286208],
    #         [0.8707655927996972, 0.1292344072003028],
    #         [0.7029225756886985, 0.2970774243113014],
    #         [0.5000000000000000, 0.5000000000000000],
    #         [0.2970774243113014, 0.7029225756886985],
    #         [0.1292344072003028, 0.8707655927996972],
    #         [0.0254460438286208, 0.9745539561713792],
    #     ]
    #     reference_weights = [
    #         0.0647424830844351,
    #         0.1398526957446383,
    #         0.1909150252525594,
    #         0.2089795918367346,
    #         0.1909150252525594,
    #         0.1398526957446383,
    #         0.0647424830844351,
    #     ]
    # elif integration_order == 8:
    #     reference_points = [
    #         [0.9801449282487681, 0.0198550717512319],
    #         [0.8983332387068134, 0.1016667612931866],
    #         [0.7627662049581645, 0.2372337950418355],
    #         [0.5917173212478249, 0.4082826787521751],
    #         [0.4082826787521751, 0.5917173212478249],
    #         [0.2372337950418355, 0.7627662049581645],
    #         [0.1016667612931866, 0.8983332387068134],
    #         [0.0198550717512319, 0.9801449282487681],
    #     ]
    #     reference_weights = [
    #         0.0506142681451882,
    #         0.1111905172266871,
    #         0.1568533229389437,
    #         0.1813418916891810,
    #         0.1813418916891810,
    #         0.1568533229389437,
    #         0.1111905172266871,
    #         0.0506142681451882,
    #     ]
    else:
        raise ValueError("quadrature order not supported")
    if quadrature_item == QuadratureItem.POINTS:
        return np.array(reference_points)
    elif quadrature_item == QuadratureItem.WEIGHTS:
        return np.array(reference_weights)
    elif quadrature_item == QuadratureItem.JACOBIAN:
        return np.array(jacobian)
    else:
        raise KeyError("either points or weights")
