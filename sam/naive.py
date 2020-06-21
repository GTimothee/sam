def naive_nb_seeks(self, sliceobj, shape, dtype, offset, order, lock):
        ''' Copies the processing of nibabel to count the number of seeks.
        '''
        if fileslice.is_fancy(sliceobj):
            raise ValueError("Cannot handle fancy indexing")
        dtype = np.dtype(dtype)
        itemsize = int(dtype.itemsize)
        segments, sliced_shape, post_slicers = fileslice.calc_slicedefs(
            sliceobj, shape, itemsize, offset, order)
        print(len(segments))
        return len(segments)


def split(self, first_dim, second_dim, third_dim, local_dir,
            filename_prefix, benchmark=False):

    """Naive strategy. Splits the 3d-image into shapes of given dimensions.

    Keyword arguments:
        first_dim, second_dim, third_dim: the desired first, second and
                                            third dimensions of the splits,
                                            respectively.
        local_dir                       : the path to the local directory
                                            in which the images will be saved
        filename_prefix                 : the filename prefix
        benchmark                       : If set to true the function will
                                            return a dictionary containing
                                            benchmark information.
    """
    try:
        if self.proxy is None:
            raise AttributeError("Cannot split an image that has not yet"
                                    "been created.")
    except AttributeError as aerr:
        print('AttributeError: ', aerr)
        sys.exit(1)

    # for benchmark, if benchmark==true
    split_read_time = 0
    split_write_time = 0
    split_seek_time = 0
    split_seek_number = 0

    num_x_iters = int(ceil(self.proxy.dataobj.shape[2] / third_dim))
    num_z_iters = int(ceil(self.proxy.dataobj.shape[1] / second_dim))
    num_y_iters = int(ceil(self.proxy.dataobj.shape[0] / first_dim))

    remainder_x = self.proxy.dataobj.shape[2] % third_dim
    remainder_z = self.proxy.dataobj.shape[1] % second_dim
    remainder_y = self.proxy.dataobj.shape[0] % first_dim

    is_rem_x = is_rem_y = is_rem_z = False

    for x in range(0, num_x_iters):

        if x == num_x_iters - 1 and remainder_x != 0:
            third_dim = remainder_x
            is_rem_x = True

        for z in range(0, num_z_iters):

            if z == num_z_iters - 1 and remainder_z != 0:
                second_dim = remainder_z
                is_rem_z = True

            for y in range(0, num_y_iters):

                if y == num_y_iters - 1 and remainder_y != 0:
                    first_dim = remainder_y
                    is_rem_y = True

                x_start = x * third_dim
                x_end = (x + 1) * third_dim

                z_start = z * second_dim
                z_end = (z + 1) * second_dim

                y_start = y * first_dim
                y_end = (y + 1) * first_dim

                # use of naive_nb_seeks to get an estimate number of seeks
                # during the reading phase
                dataobj = self.proxy.dataobj
                sliceobj = (slice(y_start, y_end),
                            slice(z_start, z_end),
                            slice(x_start, x_end))
                split_seek_number += self.naive_nb_seeks(
                                            sliceobj,
                                            dataobj.shape,
                                            dataobj.dtype,
                                            dataobj.offset,
                                            'C',
                                            None)

                # 1 seek per segment
                if benchmark:
                    t = time()
                split_array = self.proxy.dataobj[y_start:y_end,
                                                    z_start:z_end,
                                                    x_start:x_end]
                if benchmark:
                    split_read_time += time() - t

                split_image = nib.Nifti1Image(split_array, self.affine)
                imagepath = None

                # TODO: fix this so that position saved in image and not
                # in filename
                # if the remaining number of voxels does not match the
                # requested number of voxels, save the image with the given
                # filename prefix and the suffix:
                # _<x starting coordinate>_<y starting coordinate>_
                # <z starting coordinate>__rem-<x lenght>-<y-length>-
                # <z length>
                if is_rem_x or is_rem_y or is_rem_z:

                    y_length = y_end - y_start
                    z_length = z_end - z_start
                    x_length = x_end - x_start

                    imagepath = ('{0}/'
                                    '{1}_{2}_{3}_{4}__rem-{5}-{6}-{7}'
                                    '.nii.gz').format(local_dir,
                                                    filename_prefix,
                                                    y_start,
                                                    z_start,
                                                    x_start,
                                                    y_length,
                                                    z_length,
                                                    x_length)
                else:
                    imagepath = ('{0}/'
                                    '{1}_{2}_{3}_{4}'
                                    '.nii.gz').format(local_dir,
                                                    filename_prefix,
                                                    y_start,
                                                    z_start,
                                                    x_start)

                if benchmark:
                    t = time()
                nib.save(split_image, imagepath)
                if benchmark:
                    split_write_time += time()-t
                    split_seek_number += 1

                legend_path = '{0}/legend.txt'.format(local_dir)

                if benchmark:
                    t = time()
                with open(legend_path, 'a+') as im_legend:
                    im_legend.write('{0}\n'.format(imagepath))
                if benchmark:
                    split_write_time += time() - t

                is_rem_z = False
        is_rem_y = False

    if benchmark:
        return {'split_read_time': split_read_time,
                'split_write_time': split_write_time,
                'split_seek_time': split_seek_time,
                'split_nb_seeks': split_seek_number}
    else:
        return


def merge(self,
            legend,
            merge_func,
            mem=None,
            input_compressed=False,
            benchmark=False):
    """

    Keyword arguments:
    legend          : a legend containing the location of the blocks or
                        slices located within the local filesystem to use for
                        reconstruction
    merge_func      : the method in which the merging should be performed 0
                        or block_block for reading blocks and writing blocks,
                        1 or block_slice for
                        reading blocks and writing slices
                        (i.e. cluster reads), and 2 or slice_slice for
                        reading slices and writing slices
    mem             : the amount of available memory in bytes
    """

    def file_access(self):

        if self.proxy is None:
            return "w+b"
        return "r+b"

    if not self.filepath.endswith('.gz'):
        print("The reconstucted image is going to be uncompressed...")
        reconstructed = open(self.filepath, self.file_access())
    else:
        print("The reconstucted image is going to be compressed...")
        reconstructed = gzip.open(self.filepath, self.file_access())

    header_writing_time = 0
    if self.proxy is None:
        if benchmark:
            t = time()
            self.header.write_to(reconstructed)
            header_writing_time = time() - t
        else:
            self.header.write_to(reconstructed)

    m_type = Merge[merge_func]
    if input_compressed:
        print("The input splits are compressed..")

    if benchmark:
        perf_dict = self.merge_types[m_type](reconstructed,
                                                legend,
                                                mem,
                                                input_compressed,
                                                benchmark)

        # because of file opening earlier in the function
        perf_dict['merge_nb_seeks'] += 1
        perf_dict['merge_write_time'] += header_writing_time
        reconstructed.close()
        return perf_dict
    else:
        self.merge_types[m_type](reconstructed,
                                    legend,
                                    mem,
                                    input_compressed,
                                    benchmark)
        reconstructed.close()
        return