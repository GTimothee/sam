def clustered_writes(self, Y_splits, Z_splits, X_splits, out_dir,
                            mem, filename_prefix="bigbrain",
                            extension="nii", nThreads=1, benchmark=False):
    """
    Split the input image into several splits, all share with the same
    shape
    For now only supports Nifti1 images

    :param Y_splits: How many splits in Y-axis
    :param Z_splits: How many splits in Z-axis
    :param X_splits: How many splits in X-axis
    :param out_dir: Output Splits dir
    :param mem: memory load each round
    :param filename_prefix: each split's prefix filename
    :param extension: extension of each split
    :param nThreads: number of threads to trigger in each writing process
    :param benchmark: If set to true the function will return
                    a dictionary containing benchmark information.
    :return:
    """

    def get_metadata(filename_prefix, extension, out_dir):
        ''' A function.
        '''
        # calculate remainder based on the original image file
        Y_size, Z_size, X_size = self.header.get_data_shape()
        bytes_per_voxel = self.header['bitpix'] / 8
        original_img_voxels = X_size * Y_size * Z_size
        if (X_size % X_splits != 0
                or Z_size % Z_splits != 0
                or Y_size % Y_splits != 0):
            raise Exception("""There is remainder after splitting,
                    please reset the y,z,x splits""")
        x_size = X_size / X_splits
        z_size = Z_size / Z_splits
        y_size = Y_size / Y_splits

        # get all split_names and write them to the legend file
        split_names = generate_splits_name(y_size, z_size, x_size,
                                            Y_size, Z_size, X_size,
                                            out_dir,
                                            filename_prefix,
                                            extension)

        legend_file = generate_legend_file(split_names,
                                            "legend.txt",
                                            out_dir)

        # in order to reduce overhead when reading headers of splits
        # from hdfs, create a header cache in the local environment
        split_meta_cache = generate_headers_of_splits(split_names,
                                                        y_size,
                                                        z_size,
                                                        x_size,
                                                        self.header
                                                        .get_data_dtype())

        chunks_shape = (x_size, z_size, y_size)
        origarr_shape = (X_size, Z_size, Y_size)

        return (split_names,
                legend_file,
                split_meta_cache,
                bytes_per_voxel,
                chunks_shape,
                origarr_shape)

    def getNextBufferInfo(split_names,
                start_index,
                chunks_shape,
                origarr_shape,
                split_meta_cache,
                num_splits):
        ''' 
        Returns: 
            start_pos: first corner of buffer
            end_pos: end corner of buffer
            start_index: index of first block included in buffer
            end_index: index of last block included in buffer
        '''
        start_index = int(start_index)
        start_pos = pos_to_int_tuple(
                    split_ext(split_names[start_index])[0].split('_'))
        end_index = start_index + num_splits - 1
        if end_index >= len(split_names):
            end_index = len(split_names) - 1
        split_pos = pos_to_int_tuple(
                    split_ext(split_names[int(end_index)])[0].split('_'))

        x_size, z_size, y_size = chunks_shape
        end_pos = (split_pos[0] + y_size,
                    split_pos[1] + z_size,
                    split_pos[2] + x_size)
        split_pos_in_range = [pos_to_int_tuple(split_ext(x)[0].split('_'))
                                for x in split_names[
                                int(start_index):int(end_index) + 1]]

        X_size, Z_size, Y_size = origarr_shape
        end_index, end_pos = adjust_end_read(split_names, start_pos,
                                                split_pos, end_pos,
                                                start_index, end_index,
                                                split_pos_in_range, Y_size,
                                                Z_size, split_meta_cache,
                                                (y_size, z_size, x_size))
        return start_pos, end_pos, start_index, end_index

    def getRound(end_index, start_index, start_pos, split_names, chunks_shape):
        ''' A function.
        '''
        one_round_split_metadata = {}
        x_size, z_size, y_size = chunks_shape

        # for each split file
        for j in range(0, end_index - start_index + 1):  
            split_start = pos_to_int_tuple(split_ext(split_names
                                                        [start_index + j])
                                            [0].split('_'))

            # get split slices from buffer
            split_start = (split_start[0] - start_pos[0],  
                            split_start[1] - start_pos[1],
                            split_start[2] - start_pos[2])
            y_e = split_start[0] + y_size
            z_e = split_start[1] + z_size
            x_e = split_start[2] + x_size
            one_round_split_metadata[split_names[start_index + j]] = \
                (split_start[0], y_e, split_start[1], z_e,
                    split_start[2], x_e)

        caches = _split_arr(one_round_split_metadata.items(), nThreads)
        return caches

    def loop_(split_names,
                start_index,
                chunks_shape,
                origarr_shape,
                split_meta_cache,
                split_read_time,
                split_write_time,
                split_seek_time,
                split_seek_number,
                benchmark,
                num_splits):

        ''' A function.
        '''
        start_pos, end_pos, start_index, end_index = \
            getNextBufferInfo(split_names,
                    start_index,
                    chunks_shape,
                    origarr_shape,
                    split_meta_cache,
                    num_splits)

        print(("Reading from {0} at index {1} "
                "--> {2} at index {3}").format(start_pos,
                                                start_index,
                                                end_pos,
                                                end_index))

        if benchmark:
            # Compute number of seeks
            extracted_shape = (end_pos[0] - start_pos[0],
                            end_pos[1] - start_pos[1],
                            end_pos[2] - start_pos[2])
            X_size, Z_size, Y_size = origarr_shape

            if extracted_shape[0] < Y_size:
                split_seek_number += \
                    extracted_shape[1] * extracted_shape[2]
            elif extracted_shape[1] < Z_size:
                split_seek_number += extracted_shape[2]
            else:
                split_seek_number += 1

        # read buffer
        start_pos = list(map(lambda x: int(x), start_pos))
        end_pos = list(map(lambda x: int(x), end_pos))
        if benchmark:
            t = time()
            data = file_manager.read_data(start_pos, end_pos)
            t = time() - t
            print('buffer read time ', t)
            split_read_time += t
        else:
            data = file_manager.read_data(start_pos, end_pos)

        # get metadata for the split files impacted by current buffer loading
        caches = getRound(end_index,
                            start_index,
                            start_pos,
                            split_names,
                            chunks_shape)

        # write split files
        for _round in caches:
            for i in _round:
                ix = [int(x) for x in i[1]]
                split_data = data[ix[0]: ix[1], ix[2]: ix[3], ix[4]: ix[5]]

                if benchmark:
                    seek_time, write_time, seek_number = \
                        write_array_to_file(split_data,
                                            i[0],
                                            self.header_size,
                                            benchmark)

                    split_write_time += write_time
                    split_seek_time += seek_time
                    split_seek_number += seek_number
                else:
                    write_array_to_file(split_data,
                                        i[0],
                                        self.header_size,
                                        benchmark)

                start_index = end_index + 1

        if benchmark:
            return (start_index,
                    split_names,
                    split_read_time,
                    split_write_time,
                    split_seek_time,
                    split_seek_number)
        else:
            return start_index, split_names

    # ---- begin function ----
    split_read_time = 0
    split_write_time = 0
    split_seek_time = 0
    split_nb_seeks = 0

    (split_names,
        legend_file,
        split_meta_cache,
        bytes_per_voxel,
        chunks_shape,
        origarr_shape) =  \
        get_metadata(filename_prefix, extension, out_dir)

    # find number of chunks that can be loaded at a time, given "mem"(=available memory for buffer)
    start_index = end_index = 0
    mem = None if mem is not None and mem == 0 else mem
    num_splits = 0
    if mem is not None:
        num_splits = mem / \
            (bytes_per_voxel * chunks_shape[0] * chunks_shape[1] * chunks_shape[2])
    else:
        num_splits = 1
    if num_splits == 0:
        raise ValueError('Available memory is too low')

    # count 1 seek for each file open (think about disk seek)
    split_nb_seeks += len(split_names)

    # clustered writes
    while start_index < len(split_names):
        if benchmark:
            (start_index,
                split_names,
                split_read_time,
                split_write_time,
                split_seek_time,
                split_nb_seeks) = (loop_(split_names,
                                        start_index,
                                        chunks_shape,
                                        origarr_shape,
                                        split_meta_cache,
                                        split_read_time,
                                        split_write_time,
                                        split_seek_time,
                                        split_nb_seeks,
                                        benchmark,
                                        num_splits))
            print('cumul write time: ', split_write_time)
            print('cumul read time: ', split_read_time)
        else:
            start_index, split_names = (loop_(split_names,
                                                start_index,
                                                chunks_shape,
                                                origarr_shape,
                                                split_meta_cache,
                                                split_read_time,
                                                split_write_time,
                                                split_seek_time,
                                                split_nb_seeks,
                                                benchmark,
                                                num_splits))

    if benchmark:
        return {'split_read_time': split_read_time,
                'split_write_time': split_write_time,
                'split_seek_time': split_seek_time,
                'split_nb_seeks': split_nb_seeks}
    else:
        return


def _split_arr(arr, size):  # TODO comment this function
    # for python3
    arr = list(arr)
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr = arr[size:]
    arrs.append(arr)
    return arrs
    

def clustered_reads(self,
                    reconstructed,
                    legend,
                    mem,
                    input_compressed,
                    benchmark):
    """
    Reconstruct an image given a set of splits and amount of available
    memory such that it can load subset of splits into memory for faster
    processing.

    Assumes all blocks are of the same dimensions.

    Keyword arguments:
    reconstructed          : the fileobject pointing to the to-be
                                reconstructed image
    legend                 : legend containing the URIs of the splits.
                                Splits should be ordered in the way they
                                should be written (i.e. along first dimension,
                                then second, then third) for best performance
    mem                    : Amount of available memory in bytes.
                                If mem is None, it will only read one split at
                                a time
    NOTE: currently only supports nifti blocks as it uses 'bitpix' to
            determine number of bytes per voxel. Element is specific
            to nifti headers
    """

    # init
    merge_read_time = 0
    merge_seek_time = 0
    merge_nb_seeks = 0
    merge_write_time = 0
    rec_dims = self.header.get_data_shape()
    y_size = rec_dims[0]
    z_size = rec_dims[1]
    x_size = rec_dims[2]
    bytes_per_voxel = self.header['bitpix'] / 8

    # get splits
    splits = sort_split_names(legend)
    merge_nb_seeks += len(splits)  # 1 seek for each file opening

    # if a mem is inputted as 0, proceed with naive implementation
    # (same as not inputting a value for mem)
    mem = None if mem == 0 else mem
    remaining_mem = mem
    data_dict = {}
    unread_split = None
    start_index = 0
    end_index = 0

    while start_index < len(splits):
        if mem is not None:
            end_index = self.get_end_index(data_dict,
                                            remaining_mem,
                                            splits,
                                            start_index,
                                            bytes_per_voxel,
                                            y_size,
                                            z_size,
                                            x_size)
        else:
            end_index = start_index
            print("Naive reading from split index "
                    "{0} -> {1}".format(start_index, end_index))

        read_time = self.insert_elems(data_dict,
                                        splits,
                                        start_index,
                                        end_index,
                                        bytes_per_voxel,
                                        y_size,
                                        z_size,
                                        x_size,
                                        input_compressed,
                                        benchmark)
        print('read time ', read_time)
        merge_read_time += read_time

        # write
        (seek_time, write_time, num_seeks) = \
            write_dict_to_file(data_dict,
                                reconstructed,
                                bytes_per_voxel,
                                self.header_size,
                                benchmark)

        merge_seek_time += seek_time
        merge_nb_seeks += num_seeks
        merge_write_time += write_time

        print('cumul read time ', merge_read_time)
        print('cumul write time ', merge_write_time)

        remaining_mem = mem
        if start_index <= end_index:
            start_index = end_index + 1
        else:
            break

    if benchmark:
        print("Total time spent reading: ", merge_read_time)
        print("Total time spent seeking: ", merge_seek_time)
        print("Total number of seeks: ", merge_nb_seeks)
        print("Total time spent writing: ", merge_write_time)
        return {'merge_read_time': merge_read_time,
                'merge_write_time': merge_write_time,
                'merge_seek_time': merge_seek_time,
                'merge_nb_seeks': merge_nb_seeks}
    else:
        return


def get_end_index(self, data_dict, remaining_mem, splits, start_idx,
                    bytes_per_voxel, y_size, z_size, x_size):
    """
    Determine the clustered read's end index

    Keyword arguments:

    data_dict       - pre-initialized or empty (if naive) dictionary to
                        store key-value pairs representing seek position and
                        value to be written, respectively
    remaining_mem   - remaining available memory in bytes
    splits          - list of split filenames (sorted)
    start_idx       - Start position in splits for instance of clustered
                        read
    bytes_per_voxel - number of bytes for a voxel in the reconstructed
                        image
    y_size          - first dimension of reconstructed image's array size
    z_size          - second dimension of reconstructed image's array size
    x_size          - third dimension of reconstructed image's array size

    Returns: update end index of read

    """

    split_meta_cache = {}
    split_name = splits[start_idx].strip()

    split_im = start_im = Split(split_name)
    split_pos = start_pos = pos_to_int_tuple(start_im.split_pos)

    split_meta_cache[split_name] = split_im

    remaining_mem -= start_im.split_bytes

    if remaining_mem < 0:
        print("ERROR: insufficient memory provided")
        sys.exit(1)

    split_positions = []
    split_positions.append(start_pos)

    end_idx = start_idx

    for i in range(start_idx + 1, len(splits)):

        split_name = splits[i].strip()
        split_im = Split(split_name)
        split_pos = pos_to_int_tuple(split_im.split_pos)

        split_meta_cache[split_name] = split_im
        remaining_mem -= split_im.split_bytes

        if remaining_mem >= 0:
            split_positions.append(split_pos)

        end_idx = i
        if remaining_mem <= 0:
            break

    if remaining_mem < 0:
        end_idx -= 1
        split_name = splits[end_idx].strip()
        split_im = Split(split_name)
        split_pos = pos_to_int_tuple(split_im.split_pos)

    end_pos = (split_pos[0] + split_im.split_y,
                split_pos[1] + split_im.split_z,
                split_pos[2] + split_im.split_x)

    end_idx, end_pos = adjust_end_read(splits, start_pos, split_pos,
                                        end_pos, start_idx, end_idx,
                                        split_positions, y_size, z_size,
                                        split_meta_cache)
    print("Reading from position "
            "{0} (index {1}) -> {2} (index {3})".format(start_pos, start_idx,
                                                        end_pos, end_idx))
    return end_idx


def insert_elems(self, data_dict, splits, start_index, end_index,
                    bytes_per_voxel, y_size, z_size, x_size,
                    input_compressed, benchmark):
    """
    Insert contiguous strips of image data into dictionary.

    Keyword arguments:

    data_dict       - empty dictionary to store key-value pairs
                        representing seek position and value to be written,
                        respectively
    splits          - list of split filenames
    start_index     - Start position in splits for instance of clustered
                        read
    end_index       - End position in splits for instance of clustered
                        reads
    bytes_per_voxel - Amount of bytes in a voxel in the reconstructed image
    y_size          - first dimension's array size in reconstructed image
    z_size          - second dimensions's array size in reconstructed image
    x_size          - third dimensions's array size in reconstructed image

    """

    read_time = 0
    for i in range(start_index, end_index + 1):
        split_im = Split(splits[i].strip())
        split_pos = pos_to_int_tuple(split_im.split_pos)
        idx_start = 0

        if benchmark:
            t = time()
            split_data = split_im.proxy.get_data()
            read_tmp = time() - t
            read_time += read_tmp
            print('tmp read ',read_tmp)
        else:
            split_data = split_im.proxy.get_data()

        # split is a complete slice
        if split_im.split_y == y_size and split_im.split_z == z_size:
            if benchmark:
                t = time()
                data = split_data.tobytes('F')
                read_time += time() - t
            else:
                data = split_data.tobytes('F')

            key = (split_pos[0] +
                    split_pos[1] * y_size +
                    split_pos[2] * y_size * z_size)
            data_dict[key] = data

        # split is a complete row
        # WARNING: Untested
        elif split_im.split_y == y_size and split_im.split_z < z_size:
            for i in xrange(split_im.split_x):
                if benchmark:
                    t = time()
                    data = split_data[:, :, i].tobytes('F')
                    read_time += time() - t
                else:
                    data = split_data[:, :, i].tobytes('F')
                key = (split_pos[0] +
                        (split_pos[1] * y_size) +
                        (split_pos[2] + i) * y_size * z_size)
                data_dict[key] = data

        # split is an incomplete row
        else:
            for i in range(0, split_im.split_x):
                for j in range(0, split_im.split_z):
                    if benchmark:
                        t = time()
                        data = split_data[:, j, i].tobytes('F')
                        read_time += time() - t
                    else:
                        data = split_data[:, j, i].tobytes('F')
                    key = (split_pos[0] +
                            (split_pos[1] + j) * y_size +
                            (split_pos[2] + i) * y_size * z_size)
                    data_dict[key] = data

    if benchmark:
        return read_time
    else:
        return 0


def write_array_to_file(data_array, to_file, write_offset, benchmark, order='F'):
    """
    :param data_array: consists of consistent data that to bo written to the
                        file
    :param to_file: file path
    :param reconstructed: reconstructed image file to be written
    :param write_offset: file offset to be written
    :return: benchmarking params
    """
    if benchmark:
        write_time = 0
        seek_number = 0
        t = time()
        file_manager.write(to_file, data_array, write_offset, order)
        write_time += time() - t
        seek_number = 2  # 1 for opening file, 1 for seeking into the file
        seek_time = 0
        print('write time ', write_time)
        return seek_time, write_time, seek_number
    else:
        file_manager.write(to_file, data_array, write_offset, order)
        return