from h2o.mesh.gmsh.data import *


@dataclass(frozen=True)
class PhysicalEntity:
    dim: int
    tag: int
    label: str


@dataclass(frozen=True)
class DomainEntity:
    dtype: DomainType
    tag: int
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float
    phys_tags: Union[List[int], None]
    bounding_entities: Union[List[int], None]


@dataclass(frozen=True)
class DataStructure:
    num_points: int
    num_curves: int
    num_surfaces: int
    num_volumes: int
    points_data: Union[List[DomainEntity], None]
    curves_data: Union[List[DomainEntity], None]
    surfaces_data: Union[List[DomainEntity], None]
    volumes_data: Union[List[DomainEntity], None]


@dataclass(frozen=True)
class NodeEntity:
    entity_dim: int
    entity_tag: int
    tag: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class ElementEntity:
    entity_dim: int
    entity_tag: int
    element_type: int
    tag: int
    vertices_connectivity: List[int]


def get_domain_data(
    num_entities: int, line_index: int, c: List[str], domain_type: DomainType
) -> (Union[List[DomainEntity], None], int):
    if num_entities > 0:
        if domain_type != DomainType.POINT:
            domains_data = []
            for l_count in range(num_entities):
                line_index += 1
                line = c[line_index].split(" ")
                tag = int(line[0])
                min_p_x = float(line[1])
                min_p_y = float(line[2])
                min_p_z = float(line[3])
                max_p_x = float(line[4])
                max_p_y = float(line[5])
                max_p_z = float(line[6])
                num_physical_tags = int(line[7])
                offset = 7 + 1
                if num_physical_tags > 0:
                    physical_tags = []
                    for i_loc in range(num_physical_tags):
                        physical_tag = int(line[offset + i_loc])
                        physical_tags.append(physical_tag)
                else:
                    physical_tags = None
                num_bounding_curves = int(line[offset + num_physical_tags])
                offset += num_physical_tags + 1
                if num_bounding_curves > 0:
                    bounding_curves = []
                    for i_loc in range(num_bounding_curves):
                        bounding_curve = int(line[offset + i_loc])
                        bounding_curves.append(bounding_curve)
                else:
                    bounding_curves = None
                se = DomainEntity(
                    domain_type, tag, min_p_x, min_p_y, min_p_z, max_p_x, max_p_y, max_p_z, physical_tags, bounding_curves
                )
                domains_data.append(se)
        elif domain_type == DomainType.POINT:
            domains_data = []
            for p_count in range(num_entities):
                line_index += 1
                line = c[line_index].split(" ")
                tag = int(line[0])
                p_x = float(line[1])
                p_y = float(line[2])
                p_z = float(line[3])
                num_physical_tags = int(line[4])
                if num_physical_tags > 0:
                    physical_tags = []
                    for i_loc in range(num_physical_tags):
                        physical_tag = int(line[5 + i_loc])
                        physical_tags.append(physical_tag)
                else:
                    physical_tags = None
                se = DomainEntity(domain_type, tag, p_x, p_y, p_z, p_x, p_y, p_z, physical_tags, None)
                domains_data.append(se)
        else:
            raise ValueError("NO")
    else:
        domains_data = None
    return domains_data, line_index


def get_problem_euclidean_dimension(
    num_curves: int,
    num_surfaces: int,
    num_volumes: int,
) -> int:
    if num_curves == 0 and num_surfaces == 0 and num_volumes == 0:
        raise ValueError("NO")
    elif num_curves > 0 and num_surfaces == 0 and num_volumes == 0:
        return 1
    elif num_curves > 0 and num_surfaces > 0 and num_volumes == 0:
        return 2
    elif num_curves > 0 and num_surfaces > 0 and num_volumes > 0:
        return 3
    else:
        raise ValueError("NO")


def read_msh_file(msh_file_path: str) -> (DataStructure, List[NodeEntity], List[ElementEntity], int):
    with open(msh_file_path, "r") as msh_file:
        # --- READ MESH FILE
        c = msh_file.readlines()
        line_index = 4
        # --- PHYSICAL NAMES
        num_physical_entities = int(c[line_index])
        physical_entities = []
        for i in range(num_physical_entities):
            line = c[i + line_index + 1].rstrip().split(" ")
            dim = int(line[0])
            tag = int(line[1])
            label = line[2].replace("\"", "")
            pe = PhysicalEntity(dim, tag, label)
            physical_entities.append(pe)
        line_index += num_physical_entities
        offset = 3
        line_index += offset
        line = c[line_index].split(" ")
        # --- ENTITIES ENUMERATION
        num_points = int(line[0])
        num_curves = int(line[1])
        num_surfaces = int(line[2])
        num_volumes = int(line[3])
        euclidean_dimension = get_problem_euclidean_dimension(num_curves, num_surfaces, num_volumes)
        # --- POINTS, CURVES, SURFACES AND VOLUMES
        points_data, line_index = get_domain_data(num_points, line_index, c, DomainType.POINT)
        curves_data, line_index = get_domain_data(num_curves, line_index, c, DomainType.CURVE)
        surfaces_data, line_index = get_domain_data(num_surfaces, line_index, c, DomainType.SURFACE)
        volumes_data, line_index = get_domain_data(num_volumes, line_index, c, DomainType.VOLUME)
        # --- STRUCTURE
        data_structure = DataStructure(
            num_points,
            num_curves,
            num_surfaces,
            num_volumes,
            points_data,
            curves_data,
            surfaces_data,
            volumes_data,
        )
        # --- $NODES
        line_index += 3
        line = c[line_index].rstrip().split(" ")
        num_entity_blocks = int(line[0])
        num_nodes = int(line[1])
        min_node_tag = int(line[2])
        max_node_tag = int(line[3])
        # vertices = np.zeros((euclidean_dimension, num_nodes))
        node_entities = []
        # ------------------------------------------------
        for entity_count in range(num_entity_blocks):
            line_index += 1
            line = c[line_index].rstrip().split(" ")
            entity_dim = int(line[0])
            entity_tag = int(line[1])
            param = int(line[2])
            nb_nodes_in_block = int(line[3])
            nodes_tags = np.zeros((nb_nodes_in_block,), dtype=int)
            for i_loc in range(nb_nodes_in_block):
                line_index += 1
                line = c[line_index].rstrip()
                loc_tag = int(line) - 1
                nodes_tags[i_loc] = loc_tag
            for i_loc in range(nb_nodes_in_block):
                line_index += 1
                line = c[line_index].rstrip().split(" ")
                x_pos = float(line[0])
                y_pos = float(line[1])
                z_pos = float(line[2])
                # vertices[:, nodes_tags[i_loc]] = np.array([x_pos, y_pos, z_pos][:euclidean_dimension])
                node = NodeEntity(entity_dim, entity_tag, nodes_tags[i_loc], x_pos, y_pos, z_pos)
                node_entities.append(node)
        # --- $ELEMENTS
        line_index += 3
        line = c[line_index].rstrip().split(" ")
        num_entity_blocks = int(line[0])
        num_elements = int(line[1])
        min_element_tag = int(line[2])
        max_element_tag = int(line[3])
        element_entities = []
        # ------------------------------------------------
        for entity_count in range(num_entity_blocks):
            line_index += 1
            line = c[line_index].rstrip().split(" ")
            entity_dim = int(line[0])
            entity_tag = int(line[1])
            element_type = int(line[2])
            nb_elems_in_block = int(line[3])
            elems_tags = np.zeros((nb_elems_in_block,), dtype=int)
            for i_loc in range(nb_elems_in_block):
                line_index += 1
                line = c[line_index].rstrip().split(" ")
                loc_tag = int(line[0]) - 1
                elems_tags[i_loc] = loc_tag
                element_nb_nodes = get_element_data(element_type).n_nodes
                elems_vertices_connectivity = np.zeros((element_nb_nodes,), dtype=int)
                for v_count in range(element_nb_nodes):
                    elems_vertices_connectivity[v_count] = int(line[v_count + 1])
                ee = ElementEntity(entity_dim, entity_tag, element_type, loc_tag, list(elems_vertices_connectivity))
                element_entities.append(ee)
        return physical_entities, data_structure, node_entities, element_entities, euclidean_dimension


# pe, ds, nodes, element_entities, euclidean_dimension = read_msh_file("tetrahedra_1.msh")
# read_msh_file("quadrangles_0.msh")
# read_msh_file("triangles_0.msh")

# def get_msh_face_label(number_of_vertices: int) -> int:
#     """
#
#     Args:
#         number_of_vertices: the number of vertices in the mesh
#
#     Returns:
#
#     """
#     if number_of_vertices == 1:
#         return 15
#     elif number_of_vertices == 2:
#         return 1
#     elif number_of_vertices == 3:
#         return 2
#     elif number_of_vertices == 4:
#         return 3
#     elif number_of_vertices > 4:
#         raise GeometryError("POLYGONS NOT SUPPORTED YET WITH GMSH")
#     else:
#         return "c2d{}".format(number_of_vertices)

def build_mesh(msh_file_path: str):
    physical_entities, data_structure, node_entities, element_entities, euclidean_dimension = read_msh_file(msh_file_path)
    num_nodes = len(node_entities)
    vertices = np.zeros((euclidean_dimension, num_nodes), dtype=real)
    for i, node in enumerate(node_entities):
        node_array = np.array([node.x, node.y, node.z])
        # print("NODETAG : {}".format(node.tag))
        # vertices[:, i] = node_array[:euclidean_dimension]
        vertices[:, node.tag] = node_array[:euclidean_dimension]
    cells_vertices_connectivity = []
    cells_ordering = []
    cells_shape_types = []
    faces_vertices_connectivity = []
    faces_shape_types = []
    cells_faces_connectivity = []
    tags = []
    vertices_boundaries_connectivity = {}
    # --- INITIATE DICTS
    for physical_entity in physical_entities:
        if physical_entity.dim == euclidean_dimension - 1:
            vertices_boundaries_connectivity[physical_entity.label] = []
    if euclidean_dimension == 1:
        boundaries_entities = data_structure.points_data
    elif euclidean_dimension == 2:
        boundaries_entities = data_structure.curves_data
    elif euclidean_dimension == 3:
        boundaries_entities = data_structure.surfaces_data
    else:
        raise ValueError("NO")
    number_of_cells_in_mesh = 0
    for element_entity in element_entities:
        # print("elementitydim : {}".format(element_entity.entity_dim))
        if element_entity.entity_dim == euclidean_dimension:
            # print("therrree")
            number_of_cells_in_mesh += 1
            cell_vertices_connectivity = [ii-1 for ii in element_entity.vertices_connectivity]
            cells_vertices_connectivity.append(cell_vertices_connectivity)
            cell_ordering = get_element_data(element_entity.element_type).connectivity
            cells_ordering.append(cell_ordering)
            cell_shape_type = get_element_data(element_entity.element_type).shape_type
            cells_shape_types.append(cell_shape_type)
            cell_faces_connectivity = []
            for face_index, u in enumerate(cell_ordering):
                c = [cell_vertices_connectivity[k] for k in u]
                tag = "".join([str(item).zfill(20) for item in np.sort(c)])
                if tag in tags:
                    face_global_index = tags.index(tag)
                    cell_faces_connectivity.append(face_global_index)
                else:
                    tags.append(tag)
                    # face_label = get_face_label(len(c))
                    face_shape_type = get_element_data(element_entity.element_type).faces_shape_types[face_index]
                    # __check_faces_connection_item(c, face_label)
                    faces_vertices_connectivity.append(c)
                    faces_shape_types.append(face_shape_type)
                    face_global_index = len(faces_vertices_connectivity) - 1
                    cell_faces_connectivity.append(face_global_index)
            cells_faces_connectivity.append(cell_faces_connectivity)
        elif element_entity.entity_dim == euclidean_dimension - 1:
            # print(element_entity.entity_tag)
            # print(element_entity.entity_tag)
            # print(boundaries_entities)
            def get_bc_index(entity_tag: int) -> int:
                for index, bc in enumerate(boundaries_entities):
                    if bc.tag == entity_tag:
                        return index
                raise ValueError
            bc_index = get_bc_index(element_entity.entity_tag)
            physical_tags = boundaries_entities[bc_index].phys_tags
            # print(physical_tags)
            if not physical_tags is None:
                for phytag in physical_tags:
                    for physical_entity in physical_entities:
                        if physical_entity.tag == phytag:
                            for node_tag in element_entity.vertices_connectivity:
                                node_tag_shifted = node_tag - 1
                                # if not node_tag in vertices_boundaries_connectivity[physical_entity.label]:
                                if not node_tag_shifted in vertices_boundaries_connectivity[physical_entity.label]:
                                    vertices_boundaries_connectivity[physical_entity.label].append(node_tag_shifted)
    faces_boundaries_connectivity = {}
    for key, val in vertices_boundaries_connectivity.items():
        faces_boundaries_connectivity[key] = []
        for face_index, face_vertices_connectivity in enumerate(faces_vertices_connectivity):
            res = True
            for vertex_index in face_vertices_connectivity:
                if not vertex_index in val:
                    res = False
            if res:
                faces_boundaries_connectivity[key] += [face_index]
    number_of_vertices_in_mesh = num_nodes
    # number_of_cells_in_mesh
    number_of_faces_in_mesh = len(faces_vertices_connectivity)
    vertices_weights_cell_wise = np.zeros((number_of_vertices_in_mesh,), dtype=size_type)
    for cell_vertices_connectivity in cells_vertices_connectivity:
        vertices_weights_cell_wise[cell_vertices_connectivity] += np.ones(
            (len(cell_vertices_connectivity),), dtype=size_type
        )
    vertices_weights_face_wise = np.zeros((number_of_vertices_in_mesh,), dtype=size_type)
    for face_vertices_connectivity in faces_vertices_connectivity:
        vertices_weights_face_wise[face_vertices_connectivity] += np.ones(
            (len(face_vertices_connectivity),), dtype=size_type
        )
    return (
        vertices,
        euclidean_dimension,
        cells_vertices_connectivity,
        cells_ordering,
        cells_shape_types,
        faces_vertices_connectivity,
        faces_shape_types,
        cells_faces_connectivity,
        vertices_boundaries_connectivity,
        faces_boundaries_connectivity,
        number_of_vertices_in_mesh,
        number_of_cells_in_mesh,
        number_of_faces_in_mesh,
        vertices_weights_cell_wise,
        vertices_weights_face_wise
    )
    # print(physical_entities)
    # print(data_structure)
    # for key, item in vertices_boundaries_connectivity.items():
    #     print(key)
    #     print(item)
    # for key, item in faces_boundaries_connectivity.items():
    #     print(key)
    #     print(item)
    # print(faces_vertices_connectivity)


    # physical_entities
    # vertices_boundaries_connectivity =
    # faces_boundaries_connectivity =

# build_mesh("/home/dsiedel/projetcs/dev/test_msh/tetrahedra_1.msh")