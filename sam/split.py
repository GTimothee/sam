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