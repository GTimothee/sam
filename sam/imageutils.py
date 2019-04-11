import nibabel as nib
from math import ceil
from gzip import GzipFile
from io import BytesIO
import sys
import numpy as np
from time import time
import os
import logging
from enum import Enum
import gzip
import threading
from nibabel import fileslice


class Merge(Enum):
    clustered = 0
    multiple = 1


class ImageUtils:
    """ Core utility class for performing operations on images."""

    def __init__(self, filepath, first_dim=None, second_dim=None,
                 third_dim=None, dtype=None):
        """
        Keyword arguments:
            filepath                                : filepath to image
            first_dim, second_dim, third_dim        : the shape of the image.
                                                      Only required if image
                                                      needs to be generated
            dtype                                   : the numpy dtype of the
                                                      image. Only required if
                                                      image needs
                                                      to be generated
        """

        # load image
        self.filepath = filepath
        self.proxy = self.load_image(filepath)
        self.extension = split_ext(filepath)[1]

        # get header
        if self.proxy:
            self.header = self.proxy.header
            if dtype is not None:
                self.dtype = dtype
            else:
                self.dtype = self.header['datatype']
        elif first_dim and second_dim and third_dim and dtype:
            self.header = generate_header(first_dim, second_dim,
                                          third_dim, dtype)
            self.dtype = dtype
        else:
            raise ValueError('Cannot generate a header \
                                (probably missing some argument).')

        self.affine = self.header.get_best_affine()
        self.header_size = self.header.single_vox_offset

        # define merging functions
        self.merge_types = {
            Merge.clustered: self.clustered_read,
            Merge.multiple: self.multiple_reads
        }

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

    def strip_overlap(self, split_fn, split_data):

        overlaps = split_data[0]
        y_size, z_size, x_size = split_data[1]
        data = split_data[2]

        Y_size, Z_size, X_size = self.header.get_data_shape()

        start_y = overlaps[0][0]
        end_y = start_y + y_size

        start_z = overlaps[1][0]
        end_z = start_z + z_size

        start_x = overlaps[2][0]
        end_x = start_x + x_size

        data = data[start_y:end_y, start_z:end_z, start_x:end_x]

        return (split_fn, (0, (y_size, z_size, x_size), data))

    def split_clustered_writes(self, Y_splits, Z_splits, X_splits, out_dir,
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

        def threaded():
            '''st1 = time()
            for thread_round in caches:
                tds = []
                # one split's metadata triggers one thread
                for i in thread_round:
                    ix = [int(x) for x in i[1]]
                    split_data = data[ix[0]: ix[1], ix[2]: ix[3], ix[4]: ix[5]]
                    td = threading.Thread(target=write_array_to_file,
                                          args=(split_data, i[0],
                                                self.header_size))
                    td.start()
                    tds.append(td)
                    del split_data
                for t in tds:
                    t.join()
            write_time = time() - st1'''
            pass

        def file_manipulation(filename_prefix, extension, out_dir):
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
            return (split_names,
                    legend_file,
                    split_meta_cache,
                    bytes_per_voxel,
                    (x_size, z_size, y_size),
                    (X_size, Z_size, Y_size))

        def getPos(split_names,
                   start_index,
                   sizes,
                   Sizes,
                   split_meta_cache,
                   num_splits):
            ''' A function.
            '''
            start_index = int(start_index)
            start_pos = pos_to_int_tuple(
                        split_ext(split_names[start_index])[0].split('_'))
            end_index = start_index + num_splits - 1
            if end_index >= len(split_names):
                end_index = len(split_names) - 1
            split_pos = pos_to_int_tuple(
                        split_ext(split_names[int(end_index)])[0].split('_'))

            x_size, z_size, y_size = sizes
            end_pos = (split_pos[0] + y_size,
                       split_pos[1] + z_size,
                       split_pos[2] + x_size)
            split_pos_in_range = [pos_to_int_tuple(split_ext(x)[0].split('_'))
                                  for x in split_names[
                                  int(start_index):int(end_index) + 1]]

            X_size, Z_size, Y_size = Sizes
            end_index, end_pos = adjust_end_read(split_names, start_pos,
                                                 split_pos, end_pos,
                                                 start_index, end_index,
                                                 split_pos_in_range, Y_size,
                                                 Z_size, split_meta_cache,
                                                 (y_size, z_size, x_size))
            return start_pos, end_pos, start_index, end_index

        def getRound(end_index, start_index, start_pos, split_names, sizes):
            ''' A function.
            '''
            one_round_split_metadata = {}
            x_size, z_size, y_size = sizes
            for j in range(0, end_index - start_index + 1):
                split_start = pos_to_int_tuple(split_ext(split_names
                                                         [start_index + j])
                                               [0].split('_'))
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
                  sizes,
                  Sizes,
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
                getPos(split_names,
                       start_index,
                       sizes,
                       Sizes,
                       split_meta_cache,
                       num_splits)

            print(("Reading from {0} at index {1} "
                   "--> {2} at index {3}").format(start_pos,
                                                  start_index,
                                                  end_pos,
                                                  end_index))

            extracted_shape = (end_pos[0] - start_pos[0],
                               end_pos[1] - start_pos[1],
                               end_pos[2] - start_pos[2])

            X_size, Z_size, Y_size = Sizes
            if benchmark:
                if extracted_shape[0] < Y_size:
                    split_seek_number += \
                        extracted_shape[1] * extracted_shape[2]
                elif extracted_shape[1] < Z_size:
                    split_seek_number += extracted_shape[2]
                else:
                    split_seek_number += 1

            # read splits data
            data = None
            if (end_pos[0] - start_pos[0] == Y_size
                    and end_pos[1] - start_pos[1] == Z_size):
                if benchmark:
                    t = time()
                data = self.proxy.dataobj[..., start_pos[2]:end_pos[2]]
                if benchmark:
                    read_time = time() - t
                    print('read time ', read_time)
                    split_read_time += read_time
            else:
                start_pos = list(map(lambda x: int(x), start_pos))
                end_pos = list(map(lambda x: int(x), end_pos))
                if benchmark:
                    t = time()
                data = self.proxy.dataobj[start_pos[0]:end_pos[0],
                                          start_pos[1]:end_pos[1],
                                          start_pos[2]:end_pos[2]]
                if benchmark:
                    read_time = time() - t
                    print('read time ', read_time)
                    split_read_time += read_time

            # get round
            caches = getRound(end_index,
                              start_index,
                              start_pos,
                              split_names,
                              sizes)
            threaded()

            # write split files
            for round in caches:
                for i in round:
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

        # begin function
        split_read_time = 0
        split_write_time = 0
        split_seek_time = 0
        split_nb_seeks = 0

        # get metadata
        (split_names,
         legend_file,
         split_meta_cache,
         bytes_per_voxel,
         sizes,
         Sizes) =  \
            file_manipulation(filename_prefix, extension, out_dir)

        # manage memory
        start_index = end_index = 0
        mem = None if mem is not None and mem == 0 else mem
        num_splits = 0
        if mem is not None:
            num_splits = mem / \
                (bytes_per_voxel * sizes[0] * sizes[1] * sizes[2])
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
                                          sizes,
                                          Sizes,
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
                                                  sizes,
                                                  Sizes,
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

    def split_multiple_writes(self,
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

    def clustered_read(self,
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

    def load_image(self, filepath):

        """Load image into nibabel
        Keyword arguments:
        filepath            : The absolute or relative path
                              of the image
        """

        if not os.path.isfile(filepath):
            logging.warn("File does not exist. "
                         "Will only be able to reconstruct image...")
            return None

        try:
            return nib.load(filepath)
        except Exception as e:
            print("ERROR: Unable to load image into nibabel")
            sys.exit(1)

    def file_access(self):

        if self.proxy is None:
            return "w+b"
        return "r+b"


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


class Split:
    """
    It contains all the info of one split
    """

    def __init__(self, split_name, header=None):

        self.split_name = split_name

        # image is located in local file system
        if header is None:
            self.header = nib.load(split_name).header

        else:
            self.header = header
        self._get_info_from(split_name)

    def _get_info_from(self, split_name):
        self.split_pos = split_ext(split_name)[0].split('_')
        self.split_header_size = self.header.single_vox_offset
        self.bytes_per_voxel = self.header['bitpix'] / 8

        (self.split_y,
         self.split_z,
         self.split_x) = self.header.get_data_shape()

        self.split_bytes = self.bytes_per_voxel * (self.split_y *
                                                   self.split_x *
                                                   self.split_z)
        self.proxy = nib.load(split_name)


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


def _split_arr(arr, size):
    # for python3
    arr = list(arr)
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr = arr[size:]
    arrs.append(arr)
    return arrs


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
