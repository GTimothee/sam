

def multiple_writes(self,
                        Y_splits,
                        Z_splits,
                        X_splits,
                        out_dir,
                        mem,
                        filename_prefix="bigbrain",
                        extension="nii",
                        nThreads=1,
                        benchmark=False):
    """
    Split the input image into several splits,
    all share with the same shape
    For now only support .nii extension
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

    def threaded_multiple():
        '''# Using multi-threading to send data to hdfs in parallel,
        # which will parallelize writing process.
        # nThreads: number of threads that are working on writing
        # data at the same time.

        print("start {} threads to write data...".format(nThreads))

        # separate all the splits' metadata to several pieces,
        # each piece contains #nThreads splits' metadata.
        caches = _split_arr(one_round_split_metadata.items(), nThreads)

        st1 = time()

        for thread_round in caches:
            tds = []
            # one split's metadata triggers one thread
            for i in thread_round:
                ix = i[1]
                data = data_in_range[ix[0]: ix[1],
                                        ix[2]: ix[3],
                                        ix[4]: ix[5]]
                td = threading.Thread(target=write_array_to_file,
                                        args=(data, i[0], 0, benchmark))
                td.start()
                tds.append(td)
                del data
            for t in tds:
                t.join()'''
        pass

    def compute_sizes(Y_splits, Z_splits, X_splits):
        ''' A function.
        '''
        # calculate remainder based on the original image file
        Y_size, Z_size, X_size = self.header.get_data_shape()
        bytes_per_voxel = self.header['bitpix'] / 8

        if (X_size % X_splits != 0
                or Z_size % Z_splits != 0
                or Y_size % Y_splits != 0):
            raise Exception("There is remainder after splitting, \
                            please reset the y,z,x splits")
        x_size = X_size / X_splits
        z_size = Z_size / Z_splits
        y_size = Y_size / Y_splits
        return ((x_size, z_size, y_size),
                (X_size, Z_size, Y_size),
                bytes_per_voxel)

    def file_manipulation_multiple(sizes, Sizes, filename_prefix):
        ''' A function.
        '''

        x_size, z_size, y_size = sizes
        X_size, Z_size, Y_size = Sizes
        # get all split_names and write them to the legend file
        split_names = generate_splits_name(y_size, z_size, x_size, Y_size,
                                            Z_size, X_size, out_dir,
                                            filename_prefix,
                                            extension)
        generate_legend_file(split_names, "legend.txt", out_dir)

        # generate all the headers for each split
        # in order to reduce overhead when reading headers of splits
        # from hdfs, create a header cache in the local environment
        print("create split meta data dictionary...")
        split_meta_cache = generate_headers_of_splits(split_names,
                                                        y_size,
                                                        z_size,
                                                        x_size,
                                                        self.header
                                                        .get_data_dtype())

        print("Get split indexes...")
        split_indexes = get_indexes_of_all_splits(split_names,
                                                    split_meta_cache,
                                                    Y_size, Z_size)
        return split_indexes, split_names, split_meta_cache

    def get_metadata_multiple(split_indexes,
                                split_names,
                                split_meta_cache,
                                from_x_index):
        ''' A function.
        '''

        # create split metadata for all splits(position, write_range, etc.)
        one_round_split_metadata = {}
        for split_name in split_names:
            if check_in_range(next_read_index, split_indexes[split_name]):
                split = split_meta_cache[split_name]
                (X_index_min, X_index_max,
                    x_index_min, x_index_max) = \
                    extract_slices_range(split,
                                            next_read_index, Y_size,
                                            Z_size)
                y_index_min = int(split.split_pos[-3])
                z_index_min = int(split.split_pos[-2])
                y_index_max = y_index_min + split.split_y
                z_index_max = z_index_min + split.split_z
                one_round_split_metadata[split_name] = \
                    (y_index_min, y_index_max, z_index_min, z_index_max,
                        X_index_min - from_x_index,
                        X_index_max - from_x_index + 1)
        return one_round_split_metadata

    def loop_multiple(next_read_index,
                        bytes_per_voxel,
                        Sizes,
                        split_indexes,
                        split_names,
                        split_meta_cache,
                        split_read_time,
                        split_write_time,
                        split_seek_time,
                        split_seek_number,
                        benchmark):
        ''' A function.
        '''

        split_read_time = 0
        split_nb_seeks = 0

        X_size, Z_size, Y_size = Sizes
        original_img_voxels = X_size * Y_size * Z_size
        next_read_offsets = (next_read_index[0] * bytes_per_voxel,
                                next_read_index[1] * bytes_per_voxel + 1)
        print("From {} to {}".format(next_read_offsets[0],
                                        next_read_offsets[1]))
        from_x_index = index_to_voxel(next_read_index[0],
                                        Y_size, Z_size)[2]
        to_x_index = index_to_voxel(next_read_index[1] + 1,
                                    Y_size, Z_size)[2]

        # read
        print("Start reading data to memory...")
        if benchmark:
            t = time()
        data_in_range =  \
            self.proxy.dataobj[..., int(from_x_index): int(to_x_index)]
        if benchmark:
            read_time = time() -t
            print('read time ', read_time)
            split_read_time += read_time
            split_nb_seeks += 1

        one_round_split_metadata = get_metadata_multiple(split_indexes,
                                                            split_names,
                                                            split_meta_cache,
                                                            from_x_index)

        caches = _split_arr(one_round_split_metadata.items(), nThreads)
        threaded_multiple()
        for round in caches:
            for i in round:
                ix = i[1]
                ix = list(map(lambda x: int(x), ix))
                data = data_in_range[ix[0]:ix[1], ix[2]:ix[3], ix[4]:ix[5]]
                if benchmark:
                    seek_time, write_time, seek_number =  \
                        write_array_to_file(data, i[0], 0, benchmark)
                    split_write_time += write_time
                    split_seek_time += seek_time
                    split_nb_seeks += seek_number
                    print("writing data takes ", write_time)
                else:
                    write_array_to_file(data, i[0], 0, benchmark)

        next_read_index = (next_read_index[1] + 1,
                            next_read_index[1] + voxels)

        #  last write, write no more than image size
        if next_read_index[1] >= original_img_voxels:
            next_read_index = (next_read_index[0], original_img_voxels - 1)

        del caches
        del one_round_split_metadata
        del data_in_range

        if benchmark:
            return (next_read_index,
                    split_read_time,
                    split_write_time,
                    split_seek_time,
                    split_seek_number)
        else:
            return next_read_index

    # begin algorithm
    split_read_time = 0
    split_seek_time = 0
    split_write_time = 0
    split_seek_number = 0

    # preparation
    sizes, Sizes, bytes_per_voxel = compute_sizes(Y_splits,
                                                    Z_splits,
                                                    X_splits)
    X_size, Z_size, Y_size = Sizes
    original_img_voxels = X_size * Y_size * Z_size
    (split_indexes,
        split_names,
        split_meta_cache) = \
        file_manipulation_multiple(sizes,
                                    Sizes,
                                    filename_prefix)

    # drop the remainder which is less than one slice
    # if mem is less than one slice, then set mem to one slice
    mem = mem - mem % (Y_size * Z_size * bytes_per_voxel) \
        if mem >= Y_size * Z_size * bytes_per_voxel \
        else Y_size * Z_size * bytes_per_voxel
    voxels = mem // bytes_per_voxel  # get how many voxels per round
    next_read_index = (0, voxels - 1)

    while True:
        if benchmark:
            (next_read_index,
                split_read_time,
                split_write_time,
                split_seek_time,
                split_seek_number) = (loop_multiple(next_read_index,
                                                    bytes_per_voxel,
                                                    Sizes,
                                                    split_indexes,
                                                    split_names,
                                                    split_meta_cache,
                                                    split_read_time,
                                                    split_write_time,
                                                    split_seek_time,
                                                    split_seek_number,
                                                    benchmark))
        else:
            next_read_index = loop_multiple(next_read_index,
                                            bytes_per_voxel,
                                            Sizes,
                                            split_indexes,
                                            split_names,
                                            split_meta_cache,
                                            split_read_time,
                                            split_write_time,
                                            split_seek_time,
                                            split_seek_number,
                                            benchmark)
        # if write range is larger than img size, we are done
        if next_read_index[0] >= original_img_voxels:
            break

    if benchmark:
        return {'split_read_time': split_read_time,
                'split_write_time': split_write_time,
                'split_seek_time': split_seek_time,
                'split_nb_seeks': split_seek_number}
    else:
        return


def multiple_reads(self, reconstructed, legend, mem,
                    input_compressed, benchmark):
    """
    Reconstruct an image given a set of splits and amount of available
    memory.

    multiple_reads: load splits servel times to read a complete slice
    Currently it can work on random shape of splits and in unsorted order

    :param reconstructed: the fileobject pointing to the to-be
                            reconstructed image
    :param legend: containing the URIs of the splits.
    :param mem: bytes to be written into the file
    """
    Y_size, Z_size, X_size = self.header.get_data_shape()
    bytes_per_voxel = self.header['bitpix'] / 8
    header_offset = self.header.single_vox_offset
    reconstructed_img_voxels = X_size * Y_size * Z_size

    # if benchmark:
    merge_read_time = 0
    merge_seek_time = 0
    merge_write_time = 0
    merge_nb_seeks = 0

    # get how many voxels per round
    voxels = mem / bytes_per_voxel
    next_write_index = (0, voxels - 1)

    # read the headers of all the splits
    # to filter the splits out of the write range
    sorted_split_name_list = sort_split_names(legend)
    if benchmark:
        merge_nb_seeks += len(sorted_split_name_list)
    split_meta_cache = {}
    for s in sorted_split_name_list:
        split_meta_cache[s] = Split(s)
    split_indexes = get_indexes_of_all_splits(sorted_split_name_list,
                                                split_meta_cache,
                                                Y_size, Z_size)

    # Core loop
    while True:
        next_write_offsets = (next_write_index[0] * bytes_per_voxel,
                                next_write_index[1] * bytes_per_voxel + 1)
        print("**************From {} "
                "to {}*****************".format(next_write_offsets[0],
                                                next_write_offsets[1]))
        data_dict = {}
        found_first_split_in_range = False

        for split_name in sorted_split_name_list:
            in_range = check_in_range(next_write_index,
                                        split_indexes[split_name])
            if in_range:
                # READ
                found_first_split_in_range = True
                read_time_one_r = extract_rows(Split(split_name),
                                                data_dict,
                                                split_indexes[split_name],
                                                next_write_index,
                                                input_compressed, benchmark)

                if benchmark:
                    print('read time ', read_time_one_r)
                    merge_read_time += read_time_one_r

            elif not found_first_split_in_range:
                continue
            else:
                # because splits are sorted
                break

        # time to write to file
        (seek_time, write_time, seek_number) = \
            write_dict_to_file(data_dict, reconstructed,
                                bytes_per_voxel, header_offset, benchmark)

        if benchmark:
            merge_nb_seeks += seek_number
            merge_seek_time += seek_time
            merge_write_time += write_time

        next_write_index = (next_write_index[1] + 1,
                            next_write_index[1] + voxels)

        #  last write, write no more than image size
        if next_write_index[1] >= reconstructed_img_voxels:
            next_write_index = (next_write_index[0],
                                reconstructed_img_voxels - 1)

        # if write range is larger img size, we are done
        if next_write_index[0] >= reconstructed_img_voxels:
            break
        del data_dict

    # endofwhile
    if benchmark:
        print(merge_read_time, merge_write_time,
                merge_seek_time, merge_nb_seeks)
        return {'merge_read_time': merge_read_time,
                'merge_write_time': merge_write_time,
                'merge_seek_time': merge_seek_time,
                'merge_nb_seeks': merge_nb_seeks}
    else:
        return