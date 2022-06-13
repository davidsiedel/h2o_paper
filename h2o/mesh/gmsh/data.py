from h2o.h2o import *
from dataclasses import dataclass


@dataclass(frozen=True)
class ElementData:
    tag: int
    shape_type: ShapeType
    domain_type: DomainType
    d_eucli: int
    n_nodes: int
    n_faces: int
    faces_shape_types: Union[List[ShapeType], None]
    n_nodes_faces: Union[List[int], None]
    connectivity: Union[List[List[int]], None]


@dataclass(frozen=True, init=False)
class ElementDictionary:
    # --- SEG_002 :
    #                v
    #                ^
    #                |
    #                |
    #          0-----+-----1 --> u
    SEG_002: ElementData = ElementData(
        tag=1,
        shape_type=ShapeType.SEGMENT,
        domain_type=DomainType.CURVE,
        d_eucli=1,
        n_nodes=2,
        n_faces=2,
        faces_shape_types=[
            ShapeType.POINT,
            ShapeType.POINT,
        ],
        n_nodes_faces=[1, 1],
        connectivity=[
            [0],
            [1]
        ],
    )
    # --- TRI_003:
    #          v
    #          ^
    #          |
    #          2
    #          |`\
    #          |  `\
    #          |    `\
    #          |      `\
    #          |        `\
    #          0----------1 --> u
    TRI_003: ElementData = ElementData(
        tag=2,
        shape_type=ShapeType.TRIANGLE,
        domain_type=DomainType.SURFACE,
        d_eucli=2,
        n_nodes=3,
        n_faces=3,
        faces_shape_types=[
            ShapeType.SEGMENT,
            ShapeType.SEGMENT,
            ShapeType.SEGMENT,
        ],
        n_nodes_faces=[2, 2, 2],
        connectivity=[
            [0, 1],
            [1, 2],
            [2, 0],
        ],
    )
    # --- QUA_004:
    #                v
    #                ^
    #                |
    #          3-----------2
    #          |     |     |
    #          |     |     |
    #          |     +---- | --> u
    #          |           |
    #          |           |
    #          0-----------1
    QUA_004: ElementData = ElementData(
        tag=3,
        shape_type=ShapeType.QUADRANGLE,
        domain_type=DomainType.SURFACE,
        d_eucli=2,
        n_nodes=4,
        n_faces=4,
        faces_shape_types=[
            ShapeType.SEGMENT,
            ShapeType.SEGMENT,
            ShapeType.SEGMENT,
            ShapeType.SEGMENT,
        ],
        n_nodes_faces=[2, 2, 2, 2],
        connectivity=[
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
        ],
    )
    # --- TET_004:
    #                             v
    #                           .
    #                         ,/
    #                        /
    #                     2
    #                   ,/|`\
    #                 ,/  |  `\
    #               ,/    '.   `\
    #             ,/       |     `\
    #           ,/         |       `\
    #          0-----------'.--------1 --> u
    #           `\.         |      ,/
    #              `\.      |    ,/
    #                 `\.   '. ,/
    #                    `\. |/
    #                       `3
    #                          `\.
    #                             ` w
    TET_004: ElementData = ElementData(
        tag=4,
        shape_type=ShapeType.TETRAHEDRON,
        domain_type=DomainType.VOLUME,
        d_eucli=3,
        n_nodes=4,
        n_faces=4,
        faces_shape_types=[
            ShapeType.TRIANGLE,
            ShapeType.TRIANGLE,
            ShapeType.TRIANGLE,
            ShapeType.TRIANGLE,
        ],
        n_nodes_faces=[3, 3, 3, 3],
        connectivity=[
            [0, 1, 2],
            [1, 0, 3],
            [2, 1, 3],
            [0, 2, 3],
        ],
    )
    # --- HEX_008:
    #                 v
    #          3----------2
    #          |\     ^   |\
    #          | \    |   | \
    #          |  \   |   |  \
    #          |   7------+---6
    #          |   |  +-- |-- | -> u
    #          0---+---\--1   |
    #           \  |    \  \  |
    #            \ |     \  \ |
    #             \|      w  \|
    #              4----------5
    HEX_008: ElementData = ElementData(
        tag=5,
        shape_type=ShapeType.HEXAHEDRON,
        domain_type=DomainType.VOLUME,
        d_eucli=3,
        n_nodes=8,
        n_faces=6,
        faces_shape_types=[
            ShapeType.QUADRANGLE,
            ShapeType.QUADRANGLE,
            ShapeType.QUADRANGLE,
            ShapeType.QUADRANGLE,
            ShapeType.QUADRANGLE,
            ShapeType.QUADRANGLE,
        ],
        n_nodes_faces=[4, 4, 4, 4, 4, 4],
        connectivity=[
            [0, 1, 2, 3],
            [4, 7, 6, 5],
            [1, 5, 6, 2],
            [0, 3, 7, 4],
            [3, 2, 6, 7],
            [0, 4, 5, 1],
        ],
    )
    # --- PNT_001 :
    #          0 --> u
    PNT_001: ElementData = ElementData(
        tag=15,
        shape_type=ShapeType.POINT,
        domain_type=DomainType.POINT,
        d_eucli=0,
        n_nodes=1,
        n_faces=0,
        faces_shape_types=None,
        n_nodes_faces=None,
        connectivity=None
    )


def get_element_data(tag: int) -> ElementData:
    """

    Args:
        tag:

    Returns:

    """
    if tag < 1:
        raise ValueError("NO")
    elif tag == 1:
        return ElementDictionary.SEG_002
    elif tag == 2:
        return ElementDictionary.TRI_003
    elif tag == 3:
        return ElementDictionary.QUA_004
    elif tag == 4:
        return ElementDictionary.TET_004
    elif tag == 5:
        return ElementDictionary.HEX_008
    elif tag == 15:
        return ElementDictionary.PNT_001
    else:
        raise ValueError("NO")

def get_element_tag(shape: ShapeType) -> int:
    """

    Args:
        tag:

    Returns:

    """
    if shape == ShapeType.POINT:
        return 15
    elif shape == ShapeType.SEGMENT:
        return 1
    elif shape == ShapeType.TRIANGLE:
        return 2
    elif shape == ShapeType.QUADRANGLE:
        return 3
    elif shape == ShapeType.TETRAHEDRON:
        return 4
    elif shape == ShapeType.HEXAHEDRON:
        return 5
    else:
        raise ValueError("NO")