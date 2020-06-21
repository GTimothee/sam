def adjust_end_read(splits, start_pos, split_pos, end_pos, start_index,
                    end_idx, split_positions, y_size, z_size, split_meta_cache,
                    split_shape=None):
    """
    Adjusts the end split should the read not be a complete slice,
    complete row, or incomplete row

    Keyword arguments
    splits          - list of split filenames
    start_pos       - the starting position of the first split in the
                      reconstructed array
    split_pos       - the starting position of the last split in the
                      reconstructed array
    end_pos         - the end position of the last split in the
                      reconstructed array
    start_index     - the starting index of the first split read in splits
    end_idx         - the end index of the last split read in splits
    split_positions - a list of all the read split's positions
    y_size          - the first dimension of the reconstructed image's
                      array size
    z_size          - the second dimension of the reconstructed image's
                      array size
    split_shape     - shape of splits (for use with cwrites only)

    Returns: the "correct" last split, its index in splits,
             and its end position
    """

    prev_end_idx = end_idx

    # adjust end split it incomplete row spanning different slices/rows
    if (start_pos[0] > 0
            and (start_pos[2] < split_pos[2] or start_pos[1] < split_pos[1])):
        # get first row's last split
        curr_end_y = start_pos[1]

        for x in range(1, len(split_positions)):
            if split_positions[x][1] != curr_end_y:
                end_idx = start_index + x - 1
                break

    # adjust end split if splits are on different slices and slices or
    # complete rows are to be written.
    elif start_pos[2] < split_pos[2]:

        # complete slices
        if start_pos[0] == 0 and start_pos[1] == 0 and (end_pos[0] < y_size or
                                                        end_pos[1] < z_size):
            # need to find last split read before slice change

            curr_end_x = split_pos[2]
            for x in range(-2, -len(split_positions) - 1, -1):
                if split_positions[x][2] < curr_end_x:
                    end_idx = start_index + len(split_positions) + x
                    break

        # complete rows
        elif start_pos[0] == 0 and start_pos[1] > 0:

            # get first slice's last split
            curr_end_x = start_pos[2]

            for x in range(1, len(split_positions)):
                if split_positions[x][2] > curr_end_x:
                    end_idx = start_index + x - 1
                    break

    # adjust end split if splits start on the same slice but on different rows,
    # and read splits contain and incomplete row and a complete row
    elif (start_pos[2] == split_pos[2]
            and start_pos[1] < split_pos[1]
            and end_pos[0] != y_size):

        # get last split of second-to-last row
        curr_end_y = split_pos[1]
        for x in range(-2, -len(split_positions) - 1, -1):
            if split_positions[x][1] < curr_end_y:
                end_idx = start_index + len(split_positions) + x
                break
    # load new end
    if prev_end_idx != end_idx:
        try:
            split_im = split_meta_cache[splits[end_idx].strip()]
            split_pos = pos_to_int_tuple(split_im.split_pos)
            end_pos = (split_pos[0] + split_im.split_y,
                       split_pos[1] + split_im.split_z,
                       split_pos[2] + split_im.split_x)
        except Exception as e:
            split_pos = pos_to_int_tuple(split_ext(splits[end_idx].strip())
                                         [0].split('_'))
            end_pos = (split_pos[0] + split_shape[0],
                       split_pos[1] + split_shape[1],
                       split_pos[2] + split_shape[2])

    return int(end_idx), list(map(lambda x: int(x), end_pos))


def generate_splits_name(y_size, z_size, x_size, Y_size, Z_size, X_size,
                         out_dir, filename_prefix, extension):
    """
    generate all the splits' name based on the number of splits the user set
    """
    split_names = []
    for x in range(0, int(X_size), int(x_size)):
        for z in range(0, int(Z_size), int(z_size)):
            for y in range(0, int(Y_size), int(y_size)):
                split_names.append(
                    out_dir + '/' + filename_prefix +
                    '_' + str(y) + "_" + str(z) + "_" + str(x) +
                    "." + extension)
    return split_names


def generate_legend_file(split_names, legend_file_name, out_dir):
    """
    generate legend file for each all the splits
    """
    legend_file = '{0}/{1}'.format(out_dir, legend_file_name)

    with open(legend_file, 'a+') as f:
        for split_name in split_names:
            f.write('{0}\n'.format(split_name))

    return legend_file


def generate_headers_of_splits(split_names, y_size, z_size, x_size, dtype):
    """
    generate headers of each splits based on the shape and dtype
    """
    split_meta_cache = {}
    header = generate_header(y_size, z_size, x_size, dtype)

    for split_name in split_names:
        with open(split_name, 'w+b') as f:
            header.write_to(f)
        split_meta_cache[split_name] = Split(split_name, header)

    return split_meta_cache


def index_to_voxel(index, Y_size, Z_size):
    """
    index to voxel, eg. 0 -> (0,0,0).
    """
    i = index % (Y_size)
    index = index // (Y_size)
    j = index % (Z_size)
    index = index // (Z_size)
    k = index
    return (i, j, k)


def extract_slices_range(split, next_read_index, Y_size, Z_size):
    """
    extract all the slices of each split that in the read range.
    X_index: index that in original image's coordinate system
    x_index: index that in the split's coordinate system
    """
    indexes = []
    x_index_min = -1
    read_start, read_end = next_read_index
    for i in range(0, split.split_x):
        index = (int(split.split_pos[-3]) +
                 (int(split.split_pos[-2])) * Y_size +
                 (int(split.split_pos[-1]) + i) * Y_size * Z_size)
        # if split's one row is in the write range.
        if index >= read_start and index <= read_end:
            if len(indexes) == 0:
                x_index_min = i
            indexes.append(index)
        else:
            continue

    X_index_min = index_to_voxel(min(indexes), Y_size, Z_size)[2]
    X_index_max = index_to_voxel(max(indexes), Y_size, Z_size)[2]
    x_index_max = x_index_min + (X_index_max - X_index_min)

    return (X_index_min, X_index_max, x_index_min, x_index_max)


def sort_split_names(legend):
    """
    sort all the split names read from legend file
    output a sorted name list
    """
    split_position_list = []
    sorted_split_names_list = []
    split_name = ""
    with open(legend, "r") as f:
        for split_name in f:
            split_name = split_name.strip()
            split = Split(split_name)
            split_position_list.append((int(split.split_pos[-3]),
                                       (int(split.split_pos[-2])),
                                       (int(split.split_pos[-1]))))

    # sort the last element first in the tuple
    split_position_list = sorted(split_position_list, key=lambda t: t[::-1])
    for position in split_position_list:
        sorted_split_names_list \
                .append(regenerate_split_name_from_position(split_name,
                                                            position))
    return sorted_split_names_list


def regenerate_split_name_from_position(split_name, position):
    filename_prefix = split_name.strip().split('/')[-1].split('_')[0]
    filename_ext = split_name.strip().split(".", 1)[1]
    blocks_dir = split_name.strip().rsplit('/', 1)[0]
    split_name = (blocks_dir + '/' + filename_prefix + "_" +
                  str(position[0]) + "_" + str(position[1]) + "_" +
                  str(position[2]) + "." + filename_ext)
    return split_name


def extract_rows(split, data_dict, index_list, write_index,
                 input_compressed, benchmark):
    """
    extract_all the rows that in the write range,
    and write the data to a numpy array
    """
    read_time_one_r = 0
    write_index = list(map(lambda x: int(x), write_index))
    write_start, write_end = write_index
    index_list = list(map(lambda x: int(x), index_list))

    if benchmark:  # if benchmark and input_compressed:
        t = time()
        split_data = split.proxy.get_data()
        read_time_one_r += time()-t
    else:
        split_data = split.proxy.get_data()

    for n, index in enumerate(index_list):

        index_start = index
        index_end = index + split.split_y
        index_start = int(index_start)
        index_end = int(index_end)

        j = int(n % (split.split_z))
        i = int(n / (split.split_z))

        if index_start >= write_start and index_end <= write_end:
            data_bytes = split_data[..., j, i].tobytes('F')
            data_dict[index_start] = data_bytes
            '''if benchmark and not input_compressed:
                read_time_one_r += st2 - st'''

        # if split's one row's start index is in the write range,
        # but end index is outside of write range.
        elif index_start <= write_end <= index_end:
            data_bytes = split_data[: (write_end - index_start + 1), j, i] \
                .tobytes('F')
            data_dict[index_start] = data_bytes
            '''if benchmark and not input_compressed:
                read_time_one_r += st2 - st'''
        # if split's one row's end index is in the write range,
        # but start index is outside of write range.
        elif index_start <= write_start <= index_end:
            data_bytes = split_data[write_start - index_start:, j, i] \
                .tobytes('F')
            data_dict[write_start] = data_bytes
            '''if benchmark and not input_compressed:
                read_time_one_r += st2 - st'''

        # if not in the write range
        else:
            continue
    del split_data
    return read_time_one_r


def get_indexes_of_all_splits(split_names, split_meta_cache, Y_size, Z_size):
    """
    get writing offsets of all splits, add them to a dictionary
    key-> split_name
    value-> a writing offsets list
    """
    split_indexes = {}
    for split_name in split_names:
        split_name = split_name.strip()
        split = split_meta_cache[split_name]
        index_dict = get_indexes_of_split(split, Y_size, Z_size)
        split_indexes[split.split_name] = index_dict

    return split_indexes


def get_indexes_of_split(split, Y_size, Z_size):
    """
    get all the writing offset in one split

    (j,i) -> (index_start,index_end)
    """
    index_list = []
    for i in range(0, split.split_x):
        for j in range(0, split.split_z):
            # calculate the indexes (in bytes) of each tile, add all the tiles
            # in to data_dict that in the write range.
            write_index = (int(split.split_pos[-3]) +
                           (int(split.split_pos[-2]) + j) * Y_size +
                           (int(split.split_pos[-1]) + i) * Y_size * Z_size)
            index_list.append(write_index)
    return index_list


def check_in_range(next_index, index_list):
    """
    check if at least one voxel in the split in the write range
    """
    for index in index_list:
        if index >= next_index[0] and index <= next_index[1]:
            return True
    return False


def write_array_to_file(data_array, to_file, write_offset, benchmark):
    """
    :param data_array: consists of consistent data that to bo written to the
                       file
    :param to_file: file path
    :param reconstructed: reconstructed image file to be written
    :param write_offset: file offset to be written
    :return: benchmarking params
    """
    data = data_array.tobytes('F')

    # write
    if benchmark:
        write_time = 0
        seek_number = 0
        t = time()
    fd = os.open(to_file, os.O_RDWR | os.O_APPEND)
    os.pwrite(fd, data, write_offset)
    os.close(fd)
    if benchmark:
        write_time += time() - t

    del data_array
    del data

    if benchmark:
        seek_number = 2  # 1 for opening file, 1 for seeking into the file
        seek_time = 0
        print('write time ', write_time)
        return seek_time, write_time, seek_number
    else:
        return


def write_dict_to_file(data_dict,
                       to_file,
                       bytes_per_voxel,
                       header_offset,
                       benchmark):
    """
    :param data_array: consists of consistent data that to bo written to the
                       file
    :param reconstructed: reconstructed image file to be written
    :param write_offset: file offset to be written
    :return: benchmarking params
    """
    write_time = 0
    seek_number = 0

    for k in sorted(data_dict.keys()):
        seek_pos = int(header_offset + k * bytes_per_voxel)
        data_bytes = data_dict[k]

        if benchmark:
            t = time()
            os.pwrite(to_file.fileno(), data_bytes, seek_pos)
            write_time += time() - t
            seek_number += 1
        else:
            os.pwrite(to_file.fileno(), data_bytes, seek_pos)

        del data_dict[k]
        del data_bytes

    """
    if benchmark:
        t = time()
        to_file.flush()
        os.fsync(to_file)
        write_time += time() - t
    else:
        to_file.flush()
        os.fsync(to_file)
    """

    # because we don't use seek(), the seek time is
    # embedded in the writing time
    seek_time = 0
    print('write time ', write_time)
    return seek_time, write_time, seek_number


def generate_header(first_dim, second_dim, third_dim, dtype):
    # TODO: Fix header so that header information is accurate once data is
    #       filled
    # Assumes file data is 3D

    try:
        header = nib.Nifti1Header()
        header['dim'][0] = 3
        header['dim'][1] = first_dim
        header['dim'][2] = second_dim
        header['dim'][3] = third_dim
        header.set_sform(np.eye(4))
        header.set_data_dtype(dtype)

        return header

    except Exception as e:
        print("ERROR: Unable to generate header. "
              "Please verify that the dimensions and datatype are valid.")
        sys.exit(1)


def is_gzipped(filepath, buff=None):
    """Determine if image is gzipped
    Keyword arguments:
    filepath        : the absolute or relative filepath to the image
    buffer          : the bystream buffer. By default the value is None.
                      If the image is located on HDFS, it is necessary to
                      provide a buffer, otherwise, the program will terminate.
    """
    mime = magic.Magic(mime=True)
    try:
        if buff is None:
            if 'gzip' in mime.from_file(filepath):
                return True
            return False
        else:
            if 'gzip' in mime.from_buffer(buff):
                return True
            return False
    except Exception as e:
        print('ERROR: an error occured while attempting to determine if file '
              'is gzipped')
        sys.exit(1)


def is_nifti(fp):
    ext = split_ext(fp)[1]
    if '.nii' in ext:
        return True
    return False


def is_minc(fp):
    ext = split_ext(fp)[1]
    if '.mnc' in ext:
        return True
    return False


def split_ext(filepath):
    # assumes that if '.mnc' of '.nii' not in gzipped file extension,
    # all extensions have been removed
    root, ext = os.path.splitext(filepath)
    ext_1 = ext.lower()
    if '.gz' in ext_1:
        root, ext = os.path.splitext(root)
        ext_2 = ext.lower()
        if '.mnc' in ext_2 or '.nii' in ext_2:
            return root, "".join((ext_1, ext_2))
        else:
            return "".join((root, ext)), ext_1

    return root, ext_1


def pos_to_int_tuple(pos):
    return (int(pos[-3]), int(pos[-2]), int(pos[-1]))


get_bytes_per_voxel = {'uint8': np.dtype('uint8').itemsize,
                       'uint16': np.dtype('uint16').itemsize,
                       'uint32': np.dtype('uint32').itemsize,
                       'ushort': np.dtype('ushort').itemsize,
                       'int8': np.dtype('int8').itemsize,
                       'int16': np.dtype('int16').itemsize,
                       'int32': np.dtype('int32').itemsize,
                       'int64': np.dtype('int64').itemsize,
                       'float32': np.dtype('float32').itemsize,
                       'float64': np.dtype('float64').itemsize
                       }

